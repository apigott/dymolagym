within ;
model VanDerPol "Van der Pol Oscillator Model"
  Real my_time;
  Real x1(start = 1, fixed = true);
  Real x2(start = 1, fixed = true);

  Modelica.Blocks.Interfaces.RealInput u1
    annotation (Placement(transformation(extent={{-26,-20},{14,20}})));
equation
  my_time=time;
  der(x1) = x2;
  der(x2) = -x1 + (1 - x1*x1)*x2 + 0.5*u1;

  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
        coordinateSystem(preserveAspectRatio=false)),
    uses(Modelica(version="4.0.0")),
    version="1",
    conversion(noneFromVersion=""));
end VanDerPol;
