within KundurSMIB.Generation_Groups;
model Generator_CHP
  extends OpenIPSL.Electrical.Essentials.pfComponent;
  package Medium = Buildings.Media.Water;
  //Real power_demand(start=0.5);
  //Real pCOB;
  Real demand;
  Real P;
  //Modelica.Blocks.Nonlinear.SlewRateLimiter slewRateLimiter;
  Modelica.Blocks.Interfaces.RealInput u(start=2000);
  Buildings.Fluid.CHPs.ThermalElectricalFollowing eleFol(
    redeclare package Medium = Medium,
    allowFlowReversal=false,
    redeclare Buildings.Fluid.CHPs.Data.ValidationData3 per,
    m_flow_nominal=0.4,
    energyDynamics=Modelica.Fluid.Types.Dynamics.SteadyState,
    switchThermalElectricalFollowing=false,
    TEngIni=273.15 + 69.55,
    waitTime=0) "CHP unit with the electricity demand priority"
    annotation (Placement(transformation(extent={{24,-54},{44,-34}})));
  Modelica.Blocks.Sources.BooleanTable avaSig(startValue=true, table={172800})
    "Plant availability signal"
    annotation (Placement(transformation(extent={{-116,46},{-96,66}})));
  Buildings.Fluid.Sources.Boundary_pT sin1(
    redeclare package Medium = Medium,
    p(displayUnit="Pa"),
    nPorts=1) "Cooling water sink"
    annotation (Placement(transformation(extent={{84,-54},{64,-34}})));
  Buildings.Fluid.Sources.MassFlowSource_T cooWat(
    redeclare package Medium = Medium,
    use_m_flow_in=true,
    use_T_in=true,
    nPorts=1) "Cooling water source"
    annotation (Placement(transformation(extent={{-16,-54},{4,-34}})));
  Buildings.HeatTransfer.Sources.PrescribedTemperature preTem
    "Variable temperature boundary condition in Kelvin"
    annotation (Placement(transformation(extent={{-16,-94},{4,-74}})));
  Modelica.Blocks.Sources.CombiTimeTable valDat(
    tableOnFile=true,
    tableName="tab1",
    extrapolation=Modelica.Blocks.Types.Extrapolation.Periodic,
    y(unit={"W","kg/s","degC","degC","W","W","W","W","W","degC","degC","degC"}),
    offset={0,0,0,0,0,0,0,0,0,0,0,0},
    columns={2,3,4,5,6,7,8,9,10,11,12,13},
    smoothness=Modelica.Blocks.Types.Smoothness.MonotoneContinuousDerivative1,
    fileName=ModelicaServices.ExternalReferences.loadResource("modelica://Buildings/Resources/Data/Fluid/CHPs/Validation/MicroCogeneration.mos"))
  "Validation data from EnergyPlus simulation"
  annotation (Placement(transformation(extent={{-116,-4},{-96,16}})));

  Buildings.Controls.OBC.UnitConversions.From_degC TWatIn
    "Convert cooling water inlet temperature from degC to kelvin"
    annotation (Placement(transformation(extent={{-56,-54},{-36,-34}})));
  Buildings.Controls.OBC.UnitConversions.From_degC TRoo
    "Convert zone temperature from degC to kelvin"
    annotation (Placement(transformation(extent={{-56,-94},{-36,-74}})));
  OpenIPSL.Interfaces.PwPin pwPin annotation (Placement(transformation(extent={{
            62,-12},{82,8}}), iconTransformation(extent={{100,-12},{120,8}})));
equation
  connect(eleFol.port_b,sin1. ports[1])
    annotation (Line(points={{44,-44},{64,-44}},       color={0,127,255}));
  connect(avaSig.y,eleFol. avaSig) annotation (Line(points={{-95,56},{14,56},{14,
          -35},{22,-35}},       color={255,0,255}));
  connect(cooWat.ports[1],eleFol. port_a) annotation (Line(points={{4,-44},{24,-44}},
                     color={0,127,255}));
  connect(preTem.port,eleFol. TRoo) annotation (Line(points={{4,-84},{14,-84},{14,
          -51},{24,-51}},       color={191,0,0}));
  connect(valDat.y[2],cooWat. m_flow_in) annotation (Line(points={{-95,6},{-26,6},
          {-26,-36},{-18,-36}},    color={0,0,127}));
  connect(valDat.y[3],TWatIn. u) annotation (Line(points={{-95,6},{-66,6},{-66,-40},
          {-62,-40},{-62,-44},{-58,-44}},
                            color={0,0,127}));
  connect(TWatIn.y,cooWat. T_in) annotation (Line(points={{-34,-44},{-26,-44},{-26,
          -40},{-18,-40}},     color={0,0,127}));
  connect(valDat.y[4],TRoo. u) annotation (Line(points={{-95,6},{-76,6},{-76,-84},
          {-58,-84}},       color={0,0,127}));
  connect(TRoo.y,preTem. T) annotation (Line(points={{-34,-84},{-18,-84}},
          color={0,0,127}));
  //demand = slewRateLimiter.y; //valDat.y[1];
  demand = u;
  eleFol.PEleDem = demand;
  P = eleFol.PEleNet / SysData.S_b;

  [pwPin.ir; pwPin.ii] = Modelica.Math.Matrices.inv([pwPin.vr, pwPin.vi; pwPin.vi, pwPin.vr])*[P; tan(acos(0.9))*P/0.9];

  annotation (
    Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-140,-100},{100,
            80}})),
    Icon(coordinateSystem(extent={{-140,-100},{100,80}},  preserveAspectRatio=
            false),graphics={Text(
          extent={{-93,6},{-24,-12}},
          lineColor={0,0,255},
          textStyle={TextStyle.Italic},
          textString=""),Ellipse(extent={{-94,68},{98,-84}}, lineColor={28,108,
          200}),Line(points={{-28,-8},{-12,16}}, color={28,108,200}),Line(
          points={{-12,16},{14,-16},{32,10}}, color={28,108,200}),Text(
          extent={{-18,-32},{20,-64}},
          lineColor={28,108,200},
          textString="Gen1 5.2")}),
    Documentation(info="<html>
<table cellspacing=\"1\" cellpadding=\"1\" border=\"1\">
<tr>
<td><p>Reference</p></td>
<td>PSAT Manual 2.1.8</td>
</tr>
<tr>
<td><p>Last update</p></td>
<td>13/07/2015</td>
</tr>
<tr>
<td><p>Author</p></td>
<td><p>MAA Murad,SmarTS Lab, KTH Royal Institute of Technology</p></td>
</tr>
<tr>
<td><p>Contact</p></td>
<td><p><a href=\"mailto:luigiv@kth.se\">luigiv@kth.se</a></p></td>
</tr>
</table>
<p><br><span style=\"font-family: MS Shell Dlg 2;\">&LT;OpenIPSL: iTesla Power System Library&GT;</span></p>
<p><span style=\"font-family: MS Shell Dlg 2;\">Copyright 2015 RTE (France), AIA (Spain), KTH (Sweden) and DTU (Denmark)</span></p>
<ul>
<li><span style=\"font-family: MS Shell Dlg 2;\">RTE: http://www.rte-france.com/ </span></li>
<li><span style=\"font-family: MS Shell Dlg 2;\">AIA: http://www.aia.es/en/energy/</span></li>
<li><span style=\"font-family: MS Shell Dlg 2;\">KTH: https://www.kth.se/en</span></li>
<li><span style=\"font-family: MS Shell Dlg 2;\">DTU:http://www.dtu.dk/english</span></li>
</ul>
<p><span style=\"font-family: MS Shell Dlg 2;\">The authors can be contacted by email: info at itesla-ipsl dot org</span></p>
<p><span style=\"font-family: MS Shell Dlg 2;\">This package is part of the iTesla Power System Library (&QUOT;OpenIPSL&QUOT;) .</span></p>
<p><span style=\"font-family: MS Shell Dlg 2;\">The OpenIPSL is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.</span></p>
<p><span style=\"font-family: MS Shell Dlg 2;\">The OpenIPSL is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.</span></p>
<p><span style=\"font-family: MS Shell Dlg 2;\">You should have received a copy of the GNU Lesser General Public License along with the OpenIPSL. If not, see &LT;http://www.gnu.org/licenses/&GT;.</span></p>
</html>"));
end Generator_CHP;
