within ;
model NonlinearPendulumTest
  NonlinearPendulum nonlinearPendulum(c=100)
    annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
  Modelica.Blocks.Sources.Constant action(k=k);
  parameter Real k=1;
equation
  connect(action.y, nonlinearPendulum.u1)
    annotation (Line(points={{-91,0},{-0.8,0}}, color={0,0,127}));
    annotation (Placement(transformation(extent={{-112,-10},{-92,10}})),
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    uses(Modelica(version="4.0.0"), NonlinearPendulum(version="2")),
    version="1",
    conversion(noneFromVersion=""));
end NonlinearPendulumTest;
