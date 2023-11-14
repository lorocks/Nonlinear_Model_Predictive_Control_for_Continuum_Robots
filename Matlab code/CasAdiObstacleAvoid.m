clear all
close all
clc

% Import CasAdi
addpath('D:/Casadi')
import casadi.*

Ts = 0.1; % Sample time
T = 60; % Simulation time

% Obstacle Parameters
obstacle_x = 0.5;
obstacle_y = 0.5;
%obstacle_z = 0.5;
obstacle_z = 1; %Personal testing
radius_tip = 0.1;
radius_obstacle = 0.1;

% Create required references
time = zeros(1, T/Ts);
x_ref = zeros(1, T/Ts + 1) + 1;
y_ref = zeros(1, T/Ts + 1) + 1;
z_ref = zeros(1, T/Ts + 1) + 1;

for i = 0:1:T/Ts
    time(i+1) = i*Ts;
end


N = 30; % prediction horizon

% State symbols
x = MX.sym('x');
y = MX.sym('y');
z = MX.sym('z');
s = MX.sym('s');
t = MX.sym('t');
p = MX.sym('p');

% Control action symbols
s_ = MX.sym('s_');
t_ = MX.sym('t_');
p_ = MX.sym('p_');

% Initialise states and control inputs
states = [x; y; z; s; t; p]; 
controls = [s_; t_; p_];

state_length = length(states);
input_length = length(controls);

% B matrix
B = [cos(p)*(cos(t) - 1)*s_/t + s*cos(p)*(1-cos(t)-(t*sin(t)))*t_/(t*t) + -s*(cos(t)-1)*sin(p)*p_/t;
         sin(p)*(cos(t)-1)*s_/t + s*sin(p)*(1-cos(t)-(t*sin(t)))*t_/(t*t) + s*(cos(t)-1)*cos(p)*p_/t;
         sin(t)*s_/t + ((s*t*cos(t)) - sin(t))*t_/(t*t) + 0;
         s_;
         t_;
         p_];

f = Function('f',{states,controls},{B}); % State space equation
X = SX.sym('X',state_length,(N+1)); % Symbolic states over the prediction horizon
U = SX.sym('U',input_length,N); % Symbolic control actions over the prediction horizon
X_initial = SX.sym('X0', state_length);
X_reference = SX.sym('Xr', N*(state_length+input_length));
%P = SX.sym('P',state_length + N*(state_length+input_length));


J = 0; % Cost function
g = [];  % Constraints equations

% Weight matrices for state error
Q = zeros(state_length,state_length); 
Q(1,1) = 1; 
Q(2,2) = 1; 
Q(3,3) = 1; 

% Weighing matrices for control increment
R = zeros(input_length,input_length); 
R(1,1) = 0.5; 
R(2,2) = 0.5; 
R(3,3) = 0.5; 

P = [X_initial; X_reference]; % Optimisation parameters

g = [g;X(:,1)-P(1:state_length)]; % initial condition constraints
for k = 1:N
    % Compute symbolic cost function over prediction horizon
    J = J+( P((state_length+input_length)*k -2:(state_length+input_length)*k +3) - X(:,k))'*Q*( P((state_length+input_length)*k -2:(state_length+input_length)*k +3) - X(:,k)) + ...
              ( U(:,k) - P((state_length+input_length)*k +4:(state_length+input_length)*k +6))'*R*( U(:,k) - P((state_length+input_length)*k +4:(state_length+input_length)*k +6));

    next_state = X(:,k+1); % Symbolic next state
    next_plant_state = X(:,k) + (Ts*f(X(:,k), U(:,k))); % Next state found using discretised plant model
    g = [g;next_state - next_plant_state]; % Adding constraints for next states
end

% Obstacle constraints
for k = 1:N+1   % box constraints due to the map margins
    g = [g ; -sqrt((X(1,k)-obstacle_x)^2+(X(2,k)-obstacle_y)^2+(X(3,k)-obstacle_z)^2) + (radius_tip + radius_obstacle)];
end


% Reshape optimisation variables for input into optimiser
opti_variables = [reshape(X,state_length*(N+1),1);reshape(U,input_length*N,1)];

% Define optimisation problem
opti_problem = struct;
opti_problem.f = J;
opti_problem.x = opti_variables;
opti_problem.g = g;
opti_problem.p = P;

% Define optimisation options
options = struct;
options.ipopt.max_iter = 2000;
options.ipopt.print_level =0;%0,3
options.print_time = 0;
options.ipopt.acceptable_tol =1e-8;
options.ipopt.acceptable_obj_change_tol = 1e-6;

% Create solver
solver = nlpsol('solver', 'ipopt', opti_problem, options);

% Define constraints
constraints = struct;

constraints.lbg(1:state_length*(N+1)) = 0;
constraints.ubg(1:state_length*(N+1)) = 0;

constraints.lbg(state_length*(N+1)+1:state_length*(N+1)+(N+1)) = -inf;
constraints.ubg(state_length*(N+1)+1:state_length*(N+1)+(N+1)) = 0;

constraints.lbx(1:state_length:state_length*(N+1),1) = -4; %state x lower bound constraint
constraints.ubx(1:state_length:state_length*(N+1),1) = 4; %state x upper bound constraint
constraints.lbx(2:state_length:state_length*(N+1),1) = -4; %state y lower bound constraint
constraints.ubx(2:state_length:state_length*(N+1),1) = 4; %state y upper bound constraint
constraints.lbx(3:state_length:state_length*(N+1),1) = -4; %state z lower bound constraint
constraints.ubx(3:state_length:state_length*(N+1),1) = 4; %state z upper bound constraint
constraints.lbx(4:state_length:state_length*(N+1),1) = 0; %state s lower bound constraint
constraints.ubx(4:state_length:state_length*(N+1),1) = 10; %state s upper bound constraint
constraints.lbx(5:state_length:state_length*(N+1),1) = -pi; %state t lower bound constraint
constraints.ubx(5:state_length:state_length*(N+1),1) = pi; %state t upper bound constraint
constraints.lbx(6:state_length:state_length*(N+1),1) = -pi; %state p lower bound constraint
constraints.ubx(6:state_length:state_length*(N+1),1) = pi; %state p upper bound constraint

constraints.lbx(state_length*(N+1)+1:input_length:state_length*(N+1)+input_length*N,1) = 0; %s_ lower bound constraint
constraints.ubx(state_length*(N+1)+1:input_length:state_length*(N+1)+input_length*N,1) = 0.1; %s_ upper bound constraint
constraints.lbx(state_length*(N+1)+2:input_length:state_length*(N+1)+input_length*N,1) = -pi/10; %t_ lower bound constraint
constraints.ubx(state_length*(N+1)+2:input_length:state_length*(N+1)+input_length*N,1) = pi/10; %t_ upper bound constraint
constraints.lbx(state_length*(N+1)+3:input_length:state_length*(N+1)+input_length*N,1) = -pi/10; %p_ lower bound constraint
constraints.ubx(state_length*(N+1)+3:input_length:state_length*(N+1)+input_length*N,1) = pi/10; %p_ upper bound constraint

% Simulation
x0 = [0 ; 0 ; 0; 0; 0.0001; 0];% Initial start point
xs = [1; 1; 1];


x_states(:,1) = x0; % Holds all states

% Initialise variables for optimizer
u0 = zeros(N,input_length);
X0 = repmat(x0,1,N+1)';

% Start MPC
loop_num = 0;

u_cl = zeros(T/Ts + 1, 3); % Holds all optimal control actions
u_cl(1,:)=[0.05, 0, 0];

while( norm(x0(1:3, 1)-xs(1:3, 1),2) > 1e-2 && loop_num < T / Ts)
    constraints.p(1:6) = x0;
    for k = 1:N
        constraints.p((state_length+input_length)*k -2:(state_length+input_length)*k +3) = [x_ref(loop_num+1), y_ref(loop_num+1), z_ref(loop_num+1), 1, 1, 1];
        constraints.p((state_length+input_length)*k +4:(state_length+input_length)*k +6) = u_cl(loop_num+1,:);
    end

    constraints.x0  = [reshape(X0',state_length*(N+1),1);reshape(u0',input_length*N,1)];
    solution = solver('x0', constraints.x0, 'lbx', constraints.lbx, 'ubx', constraints.ubx, 'lbg', constraints.lbg, 'ubg', constraints.ubg,'p',constraints.p');
    
    u = reshape(full(solution.x(state_length*(N+1)+1:end))',input_length,N)'; % Get control actions from solver

    % Apply the control to model for new state
    [x0, u0] = model(Ts, x0, u,f);

    x_states(:,loop_num+2) = x0;

    u_cl(loop_num+2,:) = u(1,:);
    X0 = reshape(full(solution.x(1:state_length*(N+1)))',state_length,N+1)';

    X0 = [X0(2:end,:);X0(end,:)];
    loop_num = loop_num + 1;
end

figure
subplot(311)
xlabel("Time")
ylabel("s' (m/s)")
hold on
plot(time, u_cl(:,1))
grid on

subplot(312)
xlabel("Time")
ylabel("theta' (rad/s)")
hold on
plot(time, u_cl(:,2))
grid on

subplot(313)
xlabel("Time")
ylabel("phi' (rad/s)")
hold on
plot(time, u_cl(:,3))
grid on

[A, B, C]  = sphere;

figure
surf(A*radius_obstacle + obstacle_x, B*radius_obstacle + obstacle_y, C*radius_obstacle + obstacle_z)
hold on
plot3(x_states(1,:), x_states(2,:), x_states(3,:), '--') % s straight line
hold on
grid on