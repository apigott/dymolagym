within ;
model VanDerPolTest
  VanDerPol vanderpol
    annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
  parameter Real k=1;
  Modelica.Blocks.Sources.Constant action(k=k)
    annotation (Placement(transformation(extent={{-110,-10},{-90,10}})));
equation
  connect(action.y, vanderpol.u1)
    annotation (Line(points={{-89,0},{-0.6,0}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    uses(VanDerPol(version="1"), Modelica(version="4.0.0")));
end VanDerPolTest;
