#include <iostream>

#include <GL/freeglut.h>

#include "imgui.h"
#include "implot.h"
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
float translate_z = -30.0;
float translate_x, translate_y = 0.0;
bool gui_mouse_down{false};

void display();
void reshape(int w, int h);
void mouse(int glut_button, int state, int x, int y);
void motion(int x, int y);
void keyboard(unsigned char key, int x, int y);
bool initGL(int *argc, char **argv);

float imgui_co_size{100.0f};
bool imgui_draw_co{true};

std::vector<Eigen::Vector3d> ladnmarks_gt{


};

Eigen::Matrix3d findCovariance ( std::vector<Eigen::Vector3d> points ,const Eigen::Vector3d avg)
{
    Eigen::Matrix3d covariance;
    for (int x = 0; x < 3; x ++)
    {
        for (int y = 0; y < 3; y ++)
        {
            double element =0;
            for (const auto pp : points)
            {
                element += (pp(x) - avg(x)) * (pp(y) - avg(y));

            }
            covariance(x,y) = element / (points.size()-1);
        }
    };
    return covariance;
}

void draw_confusion_ellipse(const Eigen::Matrix3d& covar, Eigen::Vector3d& mean, Eigen::Vector3f color, float nstd  = 13)
{
    Eigen::LLT<Eigen::Matrix<double,3,3> > cholSolver(covar);
    Eigen::Matrix3d transform = cholSolver.matrixL();
//    std::cout << "transform " << std::endl;
//    std::cout << transform << std::endl;
    const double pi = 3.141592;
    const double di =0.02;
    const double dj =0.04;
    const double du =di*2*pi;
    const double dv =dj*pi;
    glColor3f(color.x(), color.y(),color.z());

    for (double i = 0; i < 1.0; i+=di)  //horizonal
        for (double j = 0; j < 1.0; j+=dj)  //vertical
        {
            double u = i*2*pi;      //0     to  2pi
            double v = (j-0.5)*pi;  //-pi/2 to pi/2

            const Eigen::Vector3d pp0( cos(v)* cos(u),cos(v) * sin(u),sin(v));
            const Eigen::Vector3d pp1(cos(v) * cos(u + du) ,cos(v) * sin(u + du) ,sin(v));
            const Eigen::Vector3d pp2(cos(v + dv)* cos(u + du) ,cos(v + dv)* sin(u + du) ,sin(v + dv));
            const Eigen::Vector3d pp3( cos(v + dv)* cos(u),cos(v + dv)* sin(u),sin(v + dv));
            Eigen::Vector3d tp0 = transform * (nstd*pp0) + mean;
            Eigen::Vector3d tp1 = transform * (nstd*pp1) + mean;
            Eigen::Vector3d tp2 = transform * (nstd*pp2) + mean;
            Eigen::Vector3d tp3 = transform * (nstd*pp3) + mean;

            glBegin(GL_LINE_LOOP);
            glVertex3dv(tp0.data());
            glVertex3dv(tp1.data());
            glVertex3dv(tp2.data());
            glVertex3dv(tp3.data());
            glEnd();
        }
}
void draw_confusion_ellipse2D(const Eigen::Matrix3d& covar, Eigen::Vector3d& mean, Eigen::Vector3f color, float nstd  = 3)
{
    Eigen::LLT<Eigen::Matrix<double,3,3> > cholSolver(covar);
    Eigen::Matrix3d transform = cholSolver.matrixL();
    std::cout << "transform " << std::endl;
    std::cout << transform << std::endl;
    const double pi = 3.141592;
    const double di =0.02;
    const double dj =0.04;
    const double du =di*2*pi;
    const double dv =dj*pi;
    glColor3f(color.x(), color.y(),color.z());

    for (double i = 0; i < 1.0; i+=di) { //horizonal

            double u = i*2*pi;      //0     to  2pi

            const Eigen::Vector3d pp0( cos(u), sin (u),0);
            const Eigen::Vector3d pp1( cos(u+du), sin(u+du),0);

            Eigen::Vector3d tp0 = transform * (nstd*pp0) + mean;
            Eigen::Vector3d tp1 = transform * (nstd*pp1) + mean;


            glBegin(GL_LINE_LOOP);
            glVertex3dv(tp0.data());
            glVertex3dv(tp1.data());
            glEnd();
        }
}
void draw_confusion_ellipse2D(const Eigen::Matrix2d& covar, Eigen::Vector2d& mean, Eigen::Vector3f color, float nstd  = 3)
{

    Eigen::LLT<Eigen::Matrix<double,2,2> > cholSolver(covar);
    Eigen::Matrix2d transform = cholSolver.matrixL();
//    std::cout << "transform " << std::endl;
//    std::cout << transform << std::endl;
    const double pi = 3.141592;
    const double di =0.02;
    const double dj =0.04;
    const double du =di*2*pi;
    const double dv =dj*pi;
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
int num_points(100);
Eigen::Vector3d robot_pose{0,0,0};
double v = 0;
double omega = 0;

Eigen::VectorXd mu = Eigen::Vector3d{0,0,0};
Eigen::MatrixXd cov = Eigen::Matrix3d::Identity()*1e-9;

int main (int argc, char *argv[])
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.01,0.01);

    for (int i =-1; i < 1;i++)
    {
        //ladnmarks_gt.push_back(Eigen::Vector3d(i*2, 3.5,0));
        ladnmarks_gt.push_back(Eigen::Vector3d(i*5,-3.5,0));
    }



    mu.resize(3+2*ladnmarks_gt.size());
    cov.resize(3+2*ladnmarks_gt.size(),3+2*ladnmarks_gt.size());
    cov(0,0) = 0.1;
    cov(1,1) = 0.1;
    cov(2,2) = 0.1;

    for (int i =0; i < ladnmarks_gt.size();i++)
    {
        mu[3+i*2] = ladnmarks_gt[i].x()+distribution(generator);
        mu[3+i*2+1] = ladnmarks_gt[i].y()+distribution(generator);

        cov(3+i*2,3+i*2) = 10;
        cov(3+i*2+1,3+i*2+1) = 10;
    }



    //    Eigen::VectorXd mu = Eigen::Vector3d{0,0,0};
    //    Eigen::MatrixXd cov = Eigen::Matrix3d::Identity()*0.01;
    initGL(&argc, argv);
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);
    glutMainLoop();
}




//    Jacobian of Motion Model
//
//    motion model
//    x_{t+1} = x_t+v*dt*cos(yaw)
//    y_{t+1} = y_t+v*dt*sin(yaw)
//    yaw_{t+1} = yaw_t+omega*dt
//    v_{t+1} = v{t}
//    so
//    dx/dyaw = -v*dt*sin(yaw)
//    dx/dv = dt*cos(yaw)
//    dy/dyaw = v*dt*cos(yaw)
//    dy/dv = dt*sin(yaw)


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
//    jF << 1,0, -v*sin(theta),
//            0,1, v*cos(theta),
//            0,0,1;

    jF << 1,0, 0,
            0,1, 0,
            0,0,1;


    Eigen::Matrix<double,3,3> Q;
    Q <<    0.5,   0  ,0,
            0  , 0.5 ,  0,
            0  ,0  , 0.5;
    cov.topLeftCorner(3,3) = jF * cov.topLeftCorner(3,3) * jF.transpose() +Q ;
    return cov;
}

Eigen::Vector2d fun_h (Eigen::VectorXd x, int landmark_id){

    const Eigen::Vector2d robot_pose = x.head(2);
    const Eigen::Vector2d landmark_pose {x[3+2*landmark_id],x[3+2*landmark_id+1]};
    const double robot_phi = x[2];
    const Eigen::Vector2d delta = landmark_pose - robot_pose;
    const double q = delta.squaredNorm();
    Eigen::Vector2d zp {std::sqrt(q),std::atan2(delta.y(),delta.x())-robot_phi};
    return zp;
}
Eigen::Matrix<double,2,5> getLowRangeModelJacobian(const Eigen::Vector3d& robot_pose, const Eigen::Vector2d& landmark){
    const Eigen::Vector2d delta = landmark - robot_pose.head<2>();
    const double q = delta.squaredNorm();
    Eigen::Matrix<double,2,5> H;
    H << -sqrt(q)*delta.x(), -sqrt(q)*delta.y(), 0, sqrt(q)*delta.x(), sqrt(q)*delta.y(),
            delta.y(), -delta.x(), -q, -delta.y(), delta.x();
    H = H/q;
    return H;
}


std::vector<Eigen::Vector3d> rangeModel(const Eigen::Vector3d& pose,const std::vector<Eigen::Vector3d>& landmarks, int max_range =11)
{
//
//    Eigen::Matrix4d rot_mat = Eigen::Matrix4d::Identity();
//    rot_mat << cos(robot_pose[2]),-sin(robot_pose[2]),0,robot_pose[0],
//            sin(robot_pose[2]),cos(robot_pose[2]),0,robot_pose[1],
//            0,0,1,0,
//            0,0,0,1;
//    Eigen::Matrix4d rot_mat_i = rot_mat.inverse();
    std::vector<Eigen::Vector3d> r;
    const double robot_phi = pose[2];

    Eigen::Vector3d smallest (std::numeric_limits<double>::max(),0,0);
    for (int i =0; i < landmarks.size(); i++){
        const auto &lnd = landmarks[i];
        const Eigen::Vector2d delta = lnd.head<2>() - pose.head<2>();
        const double dist = delta.norm();

        double bearing = std::atan2(delta.y(),delta.x()) - robot_phi;
        bearing = remainder(bearing, 2.0f*M_PI);
        constexpr float angle = 45.f * M_PI/180.f;
        //if (abs(bearing) < angle) {
            if (dist < max_range && dist > 0.5) {
                r.push_back(Eigen::Vector3d(dist, bearing, i));
            }
        //}
    }
    if (r.size()>0) {
        std::sort(r.begin(), r.end(),
                  [](const Eigen::Vector3d &l1, const Eigen::Vector3d &l2) { return l1.x() < l2.x(); });
        return {r[0]};
    }
    return r;
}

void display() {
    window_height = glutGet(GLUT_WINDOW_HEIGHT);
    window_width = glutGet(GLUT_WINDOW_WIDTH);
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

//    gluOrtho2D(-5,5,-1,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(translate_x, translate_y, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 0.0, 1.0);

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
        const auto mu_pred = predict_mu(mu, Eigen::Vector2d{v, omega});
        const auto cov_pred = predict_cov(mu, cov, Eigen::Vector2d{v, omega});

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

                const auto lowH = getLowRangeModelJacobian(mu_pred.head<3>(),
                                                           mu_pred.block(3 + 2 * lanmark_id, 0, 2, 1));

                std::cout << "lowH : \n" << lowH << std::endl;
                const Eigen::MatrixXd Ixd = Eigen::MatrixXd::Identity(3 + 2 * ladnmarks_gt.size(),
                                                                      3 + 2 * ladnmarks_gt.size());

                Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 3 + 2 * ladnmarks_gt.size());
                H.block<2, 3>(0, 0) = lowH.block<2, 3>(0, 0);
                H.block<2, 2>(0, 3 + 2 * lanmark_id) = lowH.block<2, 2>(0, 3);
                std::cout << "H : \n" << H << std::endl;
                std::cout << "H * cov_pred * H.t : \n" << H * cov_pred * H.transpose() << std::endl;

                const Eigen::Matrix2d R = 1e-2 * Eigen::Matrix2d::Identity();

                Eigen::MatrixXd Kalman_gain = cov_pred * H.transpose() * (H * cov_pred * H.transpose() + R).inverse();
                std::cout << "Kalman_gain : \n" << H << std::endl;
                mu = mu_pred + Kalman_gain * z_minus_zpred;
                cov = (Ixd - Kalman_gain * H) * cov_pred;
                std::cout << "cov : \n" << cov << std::endl;
                //        mu = mu_pred;
                //        cov = cov_pred;
            }
        } else {
            mu = mu_pred;
            cov = cov_pred;
        }
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
        glColor3f(1.0f, 1.0f, 0.0f);
        glVertex3f(v0.x(),v0.y(),v0.z());
        glVertex3f(v1.x(),v1.y(),v1.z());
        glEnd();
    }

    // draw robot
    glPointSize(2);
    glBegin(GL_POINTS);
    glColor3f(0.0f, .0f, 0.0f);
    glVertex3f(mu.x(),mu.y(),0);
    glEnd();



    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGLUT_NewFrame();

    ImGui::Begin("Demo Window1");

    {
        Eigen::IOFormat CleanFmt(8, 0, ", ", "\n", "[", "]");
        std::stringstream ss;
        ss << "X:\n" <<  mu.transpose().format(CleanFmt);
        ss << "\n";
        //ss << "cov:\n" << cov.format(CleanFmt);
        ImGui::Text(ss.str().c_str());
    }


    Eigen::MatrixXf ff = cov.cast<float>();
    ImPlot::BeginPlot("My Plot");
    ImPlot::PlotHeatmap("heat",ff.data(),ff.cols(),ff.rows(),ff.minCoeff(),ff.maxCoeff());
    ImPlot::EndPlot();
    for (int i =0; i < ladnmarks_gt.size(); i++){
        Eigen::Vector2d lnd_mean {mu[3+2*i],mu[3+2*i+1]};
        Eigen::Matrix2d lnd_cov = cov.block<2,2>(3+2*i,3+2*i);
        draw_confusion_ellipse2D(lnd_cov,lnd_mean, Eigen::Vector3f{0,1,0},1);
        draw_confusion_ellipse2D(lnd_cov,lnd_mean, Eigen::Vector3f{0,1,0},2);
        draw_confusion_ellipse2D(lnd_cov,lnd_mean, Eigen::Vector3f{0,1,0},3);
    }

    // draw lanmark gt
    glPointSize(6);
    glBegin(GL_POINTS);
    glColor3f(0.0f, 1.0f, 1.0f);
    for (auto & t : ladnmarks_gt){
        glVertex3dv(t.data());
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
    ImPlot::CreateContext();
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