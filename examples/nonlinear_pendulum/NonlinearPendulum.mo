within ;
model NonlinearPendulum "Nonlinear pendulum model"
  parameter Real theta0 = Modelica.Constants.pi;
  parameter Real omega0 = 20;
  parameter Real a = 2;
  parameter Real b = 3;
  parameter Real c = 0;

  Real theta(start = theta0, fixed = true);
  Real omega(start = omega0, fixed = true);

  Modelica.Blocks.Interfaces.RealInput u1
    annotation (Placement(transformation(extent={{-28,-20},{12,20}})));
initial equation
  theta = theta0;
  omega = omega0;

equation

  der(theta) = omega;
  der(omega) = - sin(theta) - b * omega + c * u1;


  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
        coordinateSystem(preserveAspectRatio=false)),
    uses(Modelica(version="3.2.3"), OpenIPSL(version="1.5.0")));
end NonlinearPendulum;
