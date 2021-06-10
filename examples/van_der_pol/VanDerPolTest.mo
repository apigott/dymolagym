within ;
model VanDerPolTest
  VanDerPol vanDerPol
    annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
  Modelica.Blocks.Noise.UniformNoise action(
    samplePeriod=0.5,
    y_min=1,
    y_max=2)
    annotation (Placement(transformation(extent={{-110,-10},{-90,10}})));
equation
  connect(action.y, vanDerPol.u1)
    annotation (Line(points={{-89,0},{-0.6,0}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    uses(VanDerPol(version="1"), Modelica(version="4.0.0")));
end VanDerPolTest;
