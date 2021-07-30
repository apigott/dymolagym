within ;
model BouncingBall "The 'classic' bouncing ball model"
  type Height=Real(unit="m");
  type Velocity=Real(unit="m/s");
  parameter Real e=0.8 "Coefficient of restitution";
  parameter Height h0=1.0 "Initial height";
  Height h "Height";
  Velocity v(start=0.0) "Velocity";
  Modelica.Blocks.Sources.Trapezoid variance(rising=k);
  parameter Real k=1;
  Real my_time;

equation
  when initial() then
    reinit(h,h0);
  end when;
  my_time = time;
  v = der(h);
  der(v) = -9.81;
  when h<0 then
    reinit(v, -e*pre(v));
  end when;
end BouncingBall;
