within ;
model NonlinearPendulum "Nonlinear pendulum model"
  Real my_time;
  parameter Real theta0 = Modelica.Constants.pi;
  parameter Real omega0 = 20;
  parameter Real a = 2;
  parameter Real b = 3;
  parameter Real c = 100;

  Real theta(start = theta0);
  Real omega(start = omega0);

  Modelica.Blocks.Interfaces.RealInput u1
    annotation (Placement(transformation(extent={{-28,-20},{12,20}})));
initial equation
  theta = theta0;
  omega = omega0;

equation
  when initial() then
    reinit(theta, theta0);
    reinit(omega, omega0);
  end when;
  my_time = time;
  der(theta) = omega;
  der(omega) = - sin(theta) - b * omega + c * u1;


  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
        coordinateSystem(preserveAspectRatio=false)),
    uses(Modelica(version="4.0.0"), OpenIPSL(version="1.5.2")),
    version="2",
    conversion(noneFromVersion="", noneFromVersion="1"));
end NonlinearPendulum;
