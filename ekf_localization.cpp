#include <iostream>
#include <GL/freeglut.h>
#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl2.h"

#include <Eigen/Dense>
#include "3rd/eigenmvn.h"
#include <random>

unsigned int window_width = 1920;
unsigned int window_height = 1080;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -100.0;
float translate_x, translate_y = 0.0;
bool gui_mouse_down{false};

void display();
void reshape(int w, int h);
void mouse(int glut_button, int state, int x, int y);
void motion(int x, int y);
void keyboard(unsigned char key, int x, int y);
bool initGL(int *argc, char **argv);

std::vector<Eigen::Vector2d> ladnmarks_gt;
std::vector<Eigen::Vector2d> trajectory_gt;
std::vector<Eigen::Vector2d> trajectory_estimate;
std::vector<Eigen::Vector2d> trajectory_free;


void draw_confusion_ellipse2D(const Eigen::Matrix2d& covar, Eigen::Vector2d& mean, Eigen::Vector3f color, float nstd  = 3)
{
    Eigen::LLT<Eigen::Matrix<double,2,2> > cholSolver(covar);
    Eigen::Matrix2d transform = cholSolver.matrixL();
    constexpr double pi = 3.141592;
    constexpr double di =0.02;
    constexpr double dj =0.04;
    constexpr double du =di*2*pi;
    constexpr double dv =dj*pi;
    glColor3f(color.x(), color.y(),color.z());

    for (double i = 0; i < 1.0; i+=di) { //horizonal

        double u = i*2*pi;      //0     to  2pi

        const Eigen::Vector2d pp0( cos(u), sin (u));
        const Eigen::Vector2d pp1( cos(u+du), sin(u+du));

        Eigen::Vector2d tp0 = transform * (nstd*pp0) + mean;
        Eigen::Vector2d tp1 = transform * (nstd*pp1) + mean;

        glBegin(GL_LINE_LOOP);
        glVertex3d(tp0.x(), tp0.y(), 0);
        glVertex3d(tp1.x(), tp1.y(), 0);
        glEnd();
    }
}

Eigen::Vector3d mean  ;
Eigen::Matrix3d covar ;
Eigen::Matrix3f covar_f ;

float imgui_Q_noise =1;
float imgui_R_noise =10;
float imgui_Odometery_err =0.001;

Eigen::Vector3d robot_pose{-35,-35,0};
double v = 0;
double omega = 0;

constexpr int STATE_SIZE=3;
Eigen::VectorXd mu = robot_pose;
Eigen::MatrixXd cov = Eigen::Matrix3d::Identity()*1e3;

Eigen::VectorXd mu_free =mu;

int main (int argc, char *argv[])
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.01,0.01);

    mu.resize(STATE_SIZE);
    cov.resize(STATE_SIZE,STATE_SIZE);

    // initialize map
    for (int i =-5; i < 5;i++)
    {
        for (int j =-5; j < 5;j++)
        {
            ladnmarks_gt.push_back(Eigen::Vector2d(i*5,j*5));
        }
    }
    initGL(&argc, argv);
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);
    glutMainLoop();
}

Eigen::VectorXd predict_mu (Eigen::VectorXd x, Eigen::Vector2d u) {
    if (u[0] == 0 && u[1] == 0)
        return x;

    Eigen::Matrix<double, 3, 2> F;
    double dt = 1;
    F << dt * cos(x[2]), 0,
            dt * sin(x[2]), 0,
            0, dt;
    x.head(3) = x.head(3) + F * u;
    return x;
}

Eigen::MatrixXd predict_cov (Eigen::VectorXd x, Eigen::MatrixXd cov, Eigen::Vector2d u){
    if (u[0] == 0 && u[1] == 0)
        return cov;
    double dt = 1;
    const double& theta = x[2];
    const double& v = u[0];

    Eigen::Matrix<double,3,3> jF;
    //jacobian of motion model
    jF << 1,0, -v*sin(theta),
          0,1, v*cos(theta),
          0,0,1;

    Eigen::Matrix<double,3,3> Q = Eigen::Matrix3d::Identity() * imgui_Q_noise;
    cov.topLeftCorner(3,3) = jF * cov.topLeftCorner(3,3) * jF.transpose() + Q ;
    return cov;
}

Eigen::Vector2d fun_h (Eigen::VectorXd x, int landmark_id){
    const Eigen::Vector2d robot_pose = x.head(2);
    const Eigen::Vector2d landmark_pose = ladnmarks_gt[landmark_id].head<2>();
    const double robot_phi = x[2];
    const Eigen::Vector2d delta = landmark_pose - robot_pose;
    const double q = delta.squaredNorm();
    Eigen::Vector2d zp {std::sqrt(q),std::atan2(delta.y(),delta.x())-robot_phi};
    return zp;
}
Eigen::Matrix<double,2,3> getJacobian(const Eigen::Vector3d& robot_pose, const Eigen::Vector2d& landmark){
    const Eigen::Vector2d delta = landmark - robot_pose.head<2>();
    const double q = delta.squaredNorm();
    Eigen::Matrix<double,2,3> H;
    H << -sqrt(q)*delta.x(), -sqrt(q)*delta.y(), 0,
            delta.y(), -delta.x(), -q;
    H = H/q;
    return H;
}

std::vector<Eigen::Vector3d> rangeModel(const Eigen::Vector3d& pose,const std::vector<Eigen::Vector2d>& landmarks, int max_range =11)
{
    const double robot_phi = pose[2];

    std::vector<Eigen::Vector3d> r;
    for (int i =0; i < landmarks.size(); i++){
        const auto &lnd = landmarks[i];
        const Eigen::Vector2d delta = lnd.head<2>() - pose.head<2>();
        const double dist = delta.norm();

        double bearing = std::atan2(delta.y(),delta.x()) - robot_phi;
        if (dist < max_range) {
            r.push_back(Eigen::Vector3d(dist, bearing, i));
        }

    }

    return r;
}

void display() {
    window_height = glutGet(GLUT_WINDOW_HEIGHT);
    window_width = glutGet(GLUT_WINDOW_WIDTH);
    // default initialization
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glEnable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat) window_width / (GLfloat) window_height, 0.01,
                   10000.0);

//    gluOrtho2D(-5,5,-1,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(translate_x, translate_y, translate_z);


    robot_pose =  predict_mu(robot_pose, Eigen::Vector2d{v,omega});

    const std::vector<Eigen::Vector3d> ranges_detections = rangeModel(robot_pose, ladnmarks_gt);

    Eigen::Matrix4d rot_mat = Eigen::Matrix4d::Identity();
    rot_mat << cos(robot_pose[2]),-sin(robot_pose[2]),0,robot_pose[0],
               sin(robot_pose[2]),cos(robot_pose[2]),0,robot_pose[1],
               0,0,1,0,0,0,0,1;
     {
         auto v0 =  Eigen::Vector4d{0,0,0,1};
         auto v1 =  Eigen::Vector4d{1,0,0,1};
         auto v2 =  Eigen::Vector4d{0,1,0,1};
         auto v3 =  Eigen::Vector4d{0,0,1,1};
         v0  = rot_mat * v0;
         v1  = rot_mat * v1;
         v2  = rot_mat * v2;
         v3  = rot_mat * v3;

         glBegin(GL_LINES);
         glColor3f(1.0f, 0.0f, 0.0f);
         glVertex3f(v0.x(),v0.y(),v0.z());
         glVertex3f(v1.x(),v1.y(),v1.z());

         glColor3f(0.0f, 1.0f, 0.0f);
         glVertex3f(v0.x(),v0.y(),v0.z());
         glVertex3f(v2.x(),v2.y(),v2.z());

         glColor3f(0.0f, 0.0f, 1.0f);
         glVertex3f(v0.x(),v0.y(),v0.z());
         glVertex3f(v3.x(),v3.y(),v3.z());
         glEnd();
    }


    if (v!=0 || omega!=0) {
        const float v_observed = v;
        const float omega_observed = omega+imgui_Odometery_err;
        const auto mu_pred = predict_mu(mu, Eigen::Vector2d{v_observed, omega_observed});
        const auto cov_pred = predict_cov(mu, cov, Eigen::Vector2d{v_observed, omega_observed});
        mu_free = predict_mu(mu_free,Eigen::Vector2d{v_observed, omega_observed});
        if (ranges_detections.size() > 0) {
            for (int i = 0; i < ranges_detections.size(); i++) {

                std::cout << "=========================" << std::endl;

                int lanmark_id = ranges_detections[i].z();
                const auto z_pred = fun_h(mu_pred, lanmark_id);

                std::cout << "mu_pred : \n" << mu_pred.transpose() << std::endl;
                std::cout << "z_pred : \n" << z_pred.transpose() << std::endl;

                const auto z = ranges_detections[i].head<2>();
                std::cout << "z : \n" << z.transpose() << std::endl;

                const auto z_minus_zpred = z - z_pred;

                const Eigen::Matrix<double,2,3> H = getJacobian(mu_pred,ladnmarks_gt[lanmark_id].head<2>());

                const Eigen::Matrix3d Ixd = Eigen::MatrixXd::Identity(STATE_SIZE,STATE_SIZE);

                std::cout << "H : \n" << H << std::endl;
                std::cout << "H * cov_pred * H.t : \n" << H * cov_pred * H.transpose() << std::endl;

                const Eigen::Matrix2d R = imgui_R_noise * Eigen::Matrix2d::Identity();

                Eigen::MatrixXd Kalman_gain = cov_pred * H.transpose() * (H * cov_pred * H.transpose() + R).inverse();
                std::cout << "Kalman_gain : \n" << H << std::endl;
                mu = mu_pred + Kalman_gain * z_minus_zpred;
                cov = (Ixd - Kalman_gain * H) * cov_pred;
                
            }
        } else {
            mu = mu_pred;
            cov = cov_pred;
        }
        trajectory_gt.push_back(robot_pose.head<2>());
        trajectory_estimate.push_back(mu.head<2>());
        trajectory_free.push_back(mu_free.head<2>());

    }

    Eigen::Vector2d tu = mu.head<2>();
    Eigen::Matrix2d tc = cov.block<2,2>(0,0);
    draw_confusion_ellipse2D(tc,tu, Eigen::Vector3f{1,0,0},1);
    draw_confusion_ellipse2D(tc,tu, Eigen::Vector3f{1,0,0},2);
    draw_confusion_ellipse2D(tc,tu, Eigen::Vector3f{1,0,0},3);

    //draw detection

    for (const auto & p : ranges_detections){
        const double bearing = p.y();
        const double dist = p.x();

        Eigen::Vector3d range_dir = {cos(bearing),sin(bearing),0};
        range_dir = dist * range_dir;
        auto v0 = Eigen::Vector4d{0,0,0,1};
        auto v1 = Eigen::Vector4d{range_dir.x(),range_dir.y(),range_dir.z(),1};
        v0  = rot_mat * v0;
        v1  = rot_mat * v1;

        glBegin(GL_LINES);
        glColor3f(0.0f, 0.0f, 0.0f);
        glVertex3f(v0.x(),v0.y(),v0.z());
        glVertex3f(v1.x(),v1.y(),v1.z());
        glEnd();
    }
    glBegin(GL_LINE_STRIP);
    glColor3f(0.5f, 0.5f, 0.5f);
    for (const auto & p : trajectory_gt) {
        glVertex3f(p.x(),p.y(),0);
    }
    glEnd();

    glBegin(GL_LINE_STRIP);
    glLineWidth(5);
    glColor3f(0.0f, 1.0f, 0.0f);
    for (const auto & p : trajectory_estimate) {
        glVertex3f(p.x()+0.01,p.y()+0.01,0);
    }
    glEnd();

    glBegin(GL_LINE_STRIP);
    glLineWidth(5);
    glColor3f(1.0f, 0, 0);
    for (const auto & p : trajectory_free) {
        glVertex3f(p.x(),p.y(),0);
    }
    glEnd();


    // draw robot
    glPointSize(2);
    glBegin(GL_POINTS);
    glColor3f(0.0f, .0f, 0.0f);
    glVertex3f(mu.x(),mu.y(),0);
    glEnd();

    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGLUT_NewFrame();

    ImGui::Begin("EKF-Localization");
    ImGui::DragFloat("Q_diag",&imgui_Q_noise,0.1f,0.0f,100.f);
    ImGui::DragFloat("R_diag",&imgui_R_noise,0.1f,0.0f,100.f);
    ImGui::DragFloat("Odometry_err", &imgui_Odometery_err, 0.01f,-0.1f,0.1f);


    {
        Eigen::IOFormat CleanFmt(8, 0, ", ", "\n", "[", "]");
        std::stringstream ss;
        ss << "X:\n" <<  mu.transpose().format(CleanFmt);
        ss << "\n";
        ss << "cov:\n" << cov.format(CleanFmt);
        ImGui::Text(ss.str().c_str());
    }

    // draw lanmark gt
    glPointSize(12);
    glBegin(GL_POINTS);
    glColor3f(0.0f, 0.0f, 1.0f);
    for (auto & t : ladnmarks_gt){
        glVertex3d(t.x(),t.y(),0);
    }
    glEnd();

    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
    glutSwapBuffers();
    glutPostRedisplay();

    v =0;
    omega = 0;
}

void mouse(int glut_button, int state, int x, int y) {
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2((float)x, (float)y);
    int button = -1;
    if (glut_button == GLUT_LEFT_BUTTON) button = 0;
    if (glut_button == GLUT_RIGHT_BUTTON) button = 1;
    if (glut_button == GLUT_MIDDLE_BUTTON) button = 2;
    if (button != -1 && state == GLUT_DOWN)
        io.MouseDown[button] = true;
    if (button != -1 && state == GLUT_UP)
        io.MouseDown[button] = false;

    if (!io.WantCaptureMouse)
    {
        if (state == GLUT_DOWN) {
            mouse_buttons |= 1 << glut_button;
        } else if (state == GLUT_UP) {
            mouse_buttons = 0;
        }
        mouse_old_x = x;
        mouse_old_y = y;
    }

}

void motion(int x, int y) {
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2((float)x, (float)y);

    if (!io.WantCaptureMouse)
    {
        float dx, dy;
        dx = (float) (x - mouse_old_x);
        dy = (float) (y - mouse_old_y);
        gui_mouse_down = mouse_buttons>0;
        if (mouse_buttons & 1) {
            rotate_x += dy * 0.2f;
            rotate_y += dx * 0.2f;
        } else if (mouse_buttons & 4) {
            translate_z += dy * 0.05f;
        } else if (mouse_buttons & 3) {
            translate_x += dx * 0.05f;
            translate_y -= dy * 0.05f;
        }
        mouse_old_x = x;
        mouse_old_y = y;
    }
    glutPostRedisplay();
}

void reshape(int w, int h) {
    glViewport(0, 0, (GLsizei) w, (GLsizei) h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat) w / (GLfloat) h, 0.01, 10000.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

bool initGL(int *argc, char **argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("perspective_camera_ba");
    glutDisplayFunc(display);
    glutMotionFunc(motion);

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glEnable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat) window_width / (GLfloat) window_height, 0.01,
                   10000.0);
    glutReshapeFunc(reshape);
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls

    ImGui::StyleColorsDark();
    ImGui_ImplGLUT_Init();
    ImGui_ImplGLUT_InstallFuncs();
    ImGui_ImplOpenGL2_Init();
    return true;
}
void keyboard(unsigned char key, int x, int y){
    key = tolower(key);

    if(key == 'w'){
        v = 0.5;
    }
    if(key == 's'){
        v = -0.5;
    }
    if(key == 'a'){
        omega = 0.25*M_PI;
    }
    if(key == 'd'){
        omega = -0.25*M_PI;
    }

}