within ;
model VanDerPolTest
  VanDerPol vanderpol
    annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
  parameter Real k=1;
  parameter Real m=1;
  Modelica.Blocks.Sources.Constant action(k=k)
    annotation (Placement(transformation(extent={{-110,-10},{-90,10}})));
  Modelica.Blocks.Sources.Constant action2(k=m)
    annotation (Placement(transformation(extent={{-12,34},{8,54}})));
equation
  connect(action.y, vanderpol.u1)
    annotation (Line(points={{-89,0},{-50,0},{-50,3.8},{-10.4,3.8}},
                                                color={0,0,127}));
  connect(action2.y, vanderpol.u2) annotation (Line(points={{9,44},{24,44},{24,6.6},
          {31.4,6.6}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    uses(VanDerPol(version="1"), Modelica(version="4.0.0"),
      IEEE9(version="3")));
end VanDerPolTest;
