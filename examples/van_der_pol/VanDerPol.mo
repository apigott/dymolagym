within ;
model VanDerPol "Van der Pol Oscillator Model"
  Real my_time;
  Real x1;
  Real x2;

  Modelica.Blocks.Interfaces.RealInput u1
    annotation (Placement(transformation(extent={{-124,18},{-84,58}})));
  OpenIPSL.Electrical.Branches.PwLine Line4_6_0(
    R=0.017,
    X=0.092,
    G=0,
    B=0.079) annotation (Placement(transformation(
        extent={{-9,-6},{9,6}},
        rotation=270,
        origin={220,-83})));
  OpenIPSL.Electrical.Branches.PwLine Line4_5_0(
    G=0,
    R=0.01,
    X=0.085,
    B=0.088) annotation (Placement(transformation(
        extent={{-9,-6},{9,6}},
        rotation=270,
        origin={62,-79})));
  OpenIPSL.Electrical.Branches.PwLine Line6_9_0(
    G=0,
    R=0.039,
    X=0.170,
    B=0.179) annotation (Placement(transformation(
        extent={{-9,-6},{9,6}},
        rotation=90,
        origin={220,-9})));
  OpenIPSL.Electrical.Branches.PwLine Line5_7_0(
    G=0,
    R=0.032,
    X=0.161,
    B=0.153) annotation (Placement(transformation(
        extent={{-9,-6},{9,6}},
        rotation=90,
        origin={62,-13})));
  OpenIPSL.Electrical.Branches.PwLine Line8_9_0(
    G=0,
    R=0.0119,
    X=0.1008,
    B=0.1045) annotation (Placement(transformation(
        extent={{-9,-6},{9,6}},
        rotation=180,
        origin={173,24})));
  OpenIPSL.Electrical.Buses.Bus B2(
    V_0=1.025,
    V_b=18,
    angle_0=9.28)
    annotation (Placement(transformation(extent={{4,14},{24,34}})));
  OpenIPSL.Electrical.Buses.Bus B7(
    V_b=230,
    V_0=1.0258,
    angle_0=3.7197)
    annotation (Placement(transformation(extent={{44,14},{64,34}})));
  OpenIPSL.Electrical.Buses.Bus B8(
    V_0=1.015882581760390,
    V_b=230,
    angle_0=0.72754) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=0,
        origin={134,24})));
  OpenIPSL.Electrical.Buses.Bus B9(
    V_b=230,
    V_0=1.0324,
    angle_0=1.9667)
    annotation (Placement(transformation(extent={{204,14},{224,34}})));
  OpenIPSL.Electrical.Buses.Bus B3(
    V_0=1.025,
    V_b=13.8,
    angle_0=4.6648)
    annotation (Placement(transformation(extent={{244,14},{264,34}})));
  OpenIPSL.Electrical.Buses.Bus B6(
    V_0=1.012654326639182,
    V_b=230,
    angle_0=-3.6874) annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=-90,
        origin={220,-46})));
  OpenIPSL.Electrical.Buses.Bus B5(
    V_0=0.995630859628167,
    V_b=230,
    angle_0=-3.9888) annotation (Placement(transformation(
        extent={{-12,-12},{12,12}},
        rotation=90,
        origin={62,-46})));
  OpenIPSL.Electrical.Buses.Bus B4(
    V_0=1.0258,
    V_b=230,
    angle_0=-2.2168) annotation (Placement(transformation(
        extent={{-12,-12},{12,12}},
        rotation=-90,
        origin={138,-98})));
  OpenIPSL.Electrical.Buses.Bus B1(
    angle_0=0,
    V_0=1.04,
    V_b=16.5) annotation (Placement(transformation(
        extent={{-12,-12},{12,12}},
        rotation=-90,
        origin={138,-128})));
  inner OpenIPSL.Electrical.SystemBase SysData(S_b=100, fn=60)
    annotation (Placement(transformation(extent={{218,-186},{318,-146}})));
  OpenIPSL.Electrical.Branches.PwLine Line7_8_0(
    R=0.0085,
    X=0.072,
    G=0,
    B=0.0745) annotation (Placement(transformation(extent={{84,18},{102,30}})));
  IEEE9.Generation_Groups.PSSEGen2gov
                                G2(
    V_b=230,
    V_0=1.04,
    angle_0=0,
    P_0=71.61309,
    Q_0=25.59159,
    height_2=0,
    tstart_2=0,
    refdisturb_2=false,
    k=0)
    annotation (Placement(transformation(extent={{-42,14},{-22,34}})));
  IEEE9.Generation_Groups.PSSEGen3gov
                                G3(
    V_b=230,
    V_0=1.025,
    angle_0=4.647661,
    P_0=85,
    Q_0=-12.50314,
    height_2=0,
    tstart_2=0,
    refdisturb_2=false,
    k=1)                annotation (Placement(transformation(
        extent={{-10,-10},{10,10}},
        rotation=180,
        origin={308,24})));
  OpenIPSL.Electrical.Branches.PSSE.TwoWindingTransformer twoWindingTransformer(
    R=0,
    X=0.0625,
    G=0,
    B=0) annotation (Placement(transformation(extent={{28,20},{40,28}})));
  OpenIPSL.Electrical.Branches.PSSE.TwoWindingTransformer twoWindingTransformer1(
    R=0,
    X=0.0586,
    G=0,
    B=0) annotation (Placement(transformation(extent={{234,20},{246,28}})));
  OpenIPSL.Electrical.Branches.PSSE.TwoWindingTransformer twoWindingTransformer2(
    R=0,
    X=0.0576,
    G=0,
    B=0) annotation (Placement(transformation(
        extent={{-6,-4},{6,4}},
        rotation=90,
        origin={138,-112})));
  IEEE9.LoadVariations.Load_variation_ramp
                                     load_B5(
    V_b=230,
    V_0=0.9975267,
    angle_0=-3.984869,
    P_0=125,
    Q_0=50,
    d_P=0.2,
    period=60)
    annotation (Placement(transformation(extent={{-2,-106},{18,-86}})));
  IEEE9.LoadVariations.Load_variation_ramp
                                     load_B8(
    V_b=230,
    V_0=1.018439,
    angle_0=0.7034081,
    P_0=100,
    Q_0=35,
    d_P=0,
    t1=0,
    period=60)  annotation (Placement(transformation(extent={{114,-30},{134,-10}})));
  IEEE9.LoadVariations.Load_variation_ramp
                                     load_B6(
    V_b=230,
    V_0=1.013555,
    angle_0=-3.685885,
    P_0=90,
    Q_0=30,
    d_P=0,
    t1=0,
    period=60)
              annotation (Placement(transformation(extent={{258,-80},{278,-60}})));
  OpenIPSL.Electrical.Machines.PSSE.GENCLS G1(
    V_b=230,
    V_0=1.04,
    P_0=71.61309,
    Q_0=25.59159,
    M_b=100,
    H=1)
    annotation (Placement(transformation(extent={{110,-176},{130,-156}})));
  Modelica.Blocks.Interfaces.RealInput u2
    annotation (Placement(transformation(extent={{294,46},{334,86}})));
equation
  when initial() then
    reinit(x1,1);
    reinit(x2,1);
  end when;
  my_time=time;
  der(x1) = x2;
  der(x2) = -x1 + (1 - x1*x1)*x2 + 0.5*u1;

  connect(Line5_7_0.n,B7. p) annotation (Line(
      points={{62,-4.9},{62,24},{54,24}},
      color={0,0,255},
      smooth=Smooth.None));
  connect(Line6_9_0.n,B9. p) annotation (Line(
      points={{220,-0.9},{220,24},{214,24}},
      color={0,0,255},
      smooth=Smooth.None));
  connect(B6.p,Line6_9_0. p) annotation (Line(
      points={{220,-46},{220,-17.1}},
      color={0,0,255},
      smooth=Smooth.None));
  connect(Line5_7_0.p,B5. p) annotation (Line(
      points={{62,-21.1},{62,-46}},
      color={0,0,255},
      smooth=Smooth.None));
  connect(Line8_9_0.p,B9. p)
    annotation (Line(points={{181.1,24},{214,24}},
                                                 color={0,0,255}));
  connect(B8.p,Line8_9_0. n)
    annotation (Line(points={{134,24},{164.9,24}},
                                                 color={0,0,255}));
  connect(B5.p,Line4_5_0. p)
    annotation (Line(points={{62,-46},{62,-70.9}}, color={0,0,255}));
  connect(B4.p,Line4_5_0. n) annotation (Line(points={{138,-98},{138,-92},{62,
          -92},{62,-87.1}},  color={0,0,255}));
  connect(B6.p,Line4_6_0. p)
    annotation (Line(points={{220,-46},{220,-74.9}},
                                                 color={0,0,255}));
  connect(B4.p,Line4_6_0. n) annotation (Line(points={{138,-98},{138,-92},{220,
          -92},{220,-91.1}},
                       color={0,0,255}));
  connect(B7.p,Line7_8_0. p)
    annotation (Line(points={{54,24},{84.9,24}},   color={0,0,255}));
  connect(B8.p,Line7_8_0. n)
    annotation (Line(points={{134,24},{101.1,24}},color={0,0,255}));
  connect(twoWindingTransformer.p,B2. p)
    annotation (Line(points={{27,24},{14,24}},     color={0,0,255}));
  connect(twoWindingTransformer.n,B7. p)
    annotation (Line(points={{41,24},{54,24}},   color={0,0,255}));
  connect(G2.pwPin,B2. p)
    annotation (Line(points={{-21,24},{14,24}},    color={0,0,255}));
  connect(B3.p,G3. pwPin)
    annotation (Line(points={{254,24},{297,24}}, color={0,0,255}));
  connect(B9.p,twoWindingTransformer1. p)
    annotation (Line(points={{214,24},{233,24}},
                                               color={0,0,255}));
  connect(twoWindingTransformer1.n,B3. p)
    annotation (Line(points={{247,24},{254,24}}, color={0,0,255}));
  connect(twoWindingTransformer2.n,B4. p)
    annotation (Line(points={{138,-105},{138,-98}},
                                                 color={0,0,255}));
  connect(B1.p,twoWindingTransformer2. p)
    annotation (Line(points={{138,-128},{138,-119}},      color={0,0,255}));
  connect(load_B5.p,B5. p) annotation (Line(points={{8,-86},{8,-58},{62,-58},{
          62,-46}},  color={0,0,255}));
  connect(load_B8.p,B8. p) annotation (Line(points={{124,-10},{122,-10},{122,24},
          {134,24}},color={0,0,255}));
  connect(load_B6.p,B6. p)
    annotation (Line(points={{268,-60},{220,-60},{220,-46}},
                                                        color={0,0,255}));
  connect(G1.p,B1. p)
    annotation (Line(points={{130,-166},{138,-166},{138,-128}},
                                                             color={0,0,255}));
  connect(u1, G2.u) annotation (Line(points={{-104,38},{-23.4,38},{-23.4,33.6}},
        color={0,0,127}));
  connect(u2, G3.u) annotation (Line(points={{314,66},{328,66},{328,14.2},{
          299.4,14.2}}, color={0,0,127}));
  annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
        coordinateSystem(preserveAspectRatio=false)),
    uses(Modelica(version="4.0.0"),
      OpenIPSL(version="1.5.2"),
      IEEE9(version="3")),
    version="1",
    conversion(noneFromVersion=""));
end VanDerPol;
