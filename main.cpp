#include <iostream>

#include <GL/freeglut.h>

#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl2.h"
#include <Eigen/Dense>
#include "3rd/eigenmvn.h"
#include <random>

const unsigned int window_width = 1920;
const unsigned int window_height = 1080;
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
bool initGL(int *argc, char **argv);

float imgui_co_size{100.0f};
bool imgui_draw_co{true};

std::vector<Eigen::Vector3d> dataset;

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

void draw_confusion_ellipse(const Eigen::Matrix3d& covar, Eigen::Vector3d& mean, Eigen::Vector3f color, float nstd  = 3)
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
Eigen::Vector3d mean  ;
Eigen::Matrix3d covar ;
Eigen::Matrix3f covar_f ;
int num_points(100);
int main (int argc, char *argv[])
{
    covar = Eigen::Matrix3d::Identity();
    mean = Eigen::Vector3d{0.01,0.001,0.01};
    covar <<2,.2, .4,
           .2, 1, .2,
           .4, .2, 3;
    covar_f = covar.cast<float>();
    Eigen::EigenMultivariateNormal<double> normX_solver(mean,covar);
    for (int i =-0; i < num_points; i ++)
    {
        dataset.emplace_back(normX_solver.samples(1));
    }
    initGL(&argc, argv);
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutMainLoop();
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(translate_x, translate_y, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 0.0, 1.0);


    Eigen::Matrix3d covariance_measured = findCovariance(dataset, mean);
    glLineWidth(1);
    draw_confusion_ellipse(covariance_measured,mean, {0,1,1});
    draw_confusion_ellipse(covar,mean, {1,0,0});

    glLineWidth(5);
    draw_confusion_ellipse2D(covariance_measured,mean, {0,1,1});
    draw_confusion_ellipse2D(covar,mean, {1,0,0});

    if (imgui_draw_co) {
        glBegin(GL_LINES);
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(imgui_co_size, 0.0f, 0.0f);

        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, imgui_co_size, 0.0f);

        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, imgui_co_size);
        glEnd();
    }
    glPointSize(2);
    glBegin(GL_POINTS);

    glColor3f(0.0f, 1.0f, 0.0f);
    for (auto &d : dataset){
        glVertex3f(d.x(), d.y(), d.z());
    }

    glEnd();

    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGLUT_NewFrame();
    ImGui::Begin("Demo Window1");
    ImGui::Text("Covariance");

    ImGui::InputFloat3("c0",covar_f.data());
    ImGui::InputFloat3("c1",covar_f.data()+3);
    ImGui::InputFloat3("c2",covar_f.data()+6);

    ImGui::InputInt("num_points",&num_points);
    if(ImGui::Button("randomize"))
    {
        covar = covar_f.cast<double>();
        dataset.clear();
        Eigen::EigenMultivariateNormal<double> normX_solver(mean,covar, false, time(0));
        for (int i =-0; i < num_points; i ++)
        {
            dataset.emplace_back(normX_solver.samples(1));
        }
    }

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
    glutSwapBuffers();
    glutPostRedisplay();

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