# How to read

## Pseudo example

```java
// comment to human reader, machine interpreter ignore these lines.
construction 1 : premise 1
construction 2 : premise 2
assumption 1 :  # empty premise
assumption 2 :
relation to prove : ?  
// question mark marks the end of problem
// and denote the beginning of solution.
{1}
// solution block 1
// immediately follows the question.
intermediate conclusion 1 : premise 1
intermediate conclusion 2 : premise 2
{2:1 X:Y N:M}
// copy solution block 1, except exchange variables X <-> Y and M <-> N
{3}
// another solution block
last conclusion : last premise
// last conclusion should be equivalent to "relation to prove"
```

## Notation convention

```java
point P
point Q
segment PQ  // pair of points
angle PQR  // triple of points
triangle <ABC>  // triangle name enclosed within <>
line pq  // pq is the line connecting P and Q, its name is PQ.lower()
pq \\ ab  // pq is parallel to ab
circle [O]  // circle name is always enclosed within square brackets []
ab_hp[C]  // the halfplane divided by line ab that contains C
ab_hp[!C]  // the halfplane divided by line ab that does NOT contain C
```

# 1. IMO 2019 SL G1

Let ABC be a triangle. Circle Γ passes through A, meets segments AB and AC again at points D and E respectively, and intersect segment BC at F and G such that F lies between B and G. The tangent to circle BDF at F and the tangent to circle CEG at G meet at point T. Suppose that points A and T are distinct. Prove that line AT is parallel to BC.

```java
normal triangle <ABC> :
point F : between B C
point G : between F C
circle [T] : contains A F G
point D : intersect [T] AB
point E : intersect [T] AC
circle [Of] : contains D B F
line ft : tangent [Of] F
circle [Og] : contains C E G
line gt : tangent [Og] G
point T : intersect ft gt
distinct A T :
at \\ bc : ?
{1}
TFB = FDA : tangent ft [Of]
FDA = CGA : [T] contains A D F G
# TFB = CGA
{2:1 F:G D:E C:B}
TGC = GEA : tangent gt [Og]
GEA = BFA : [T] contains A E G F
# TGB = CFA
{3}
<FGA> = <GFT> : ASA FG = GF
[T] contains T : FAG = FTG
AFG = ATG : [T] contains A F T G
at \\ fg : ATG = TGF
```

# 2. IMO 2019 SL G2

Let ABC be an acute-angled triangle and let D, E, and F be the feet of altitudes from A, B, and C to sides BC, CA, and AB, respectively. Denote by ωB and ωC the incircles of triangles BDF and CDE, and let these circles be tangent to segments DF and DE at M and N, respectively. Let line MN meet circles ωB and ωC again at P != M and Q != N, respectively. Prove that MP = NQ.

```java
acute triangle <ABC> :
line ad : perp A bc
point D : intersect ad BC
line be : perp B ca
point E : intersect be AC
line cf : perp C AB
point F : intersect cf AB
point Ob , circle [Ob] : incircle <BDF>
point M : intersect [Ob] FD
point Oc , circle [Oc] : incircle <CDE>
point N : intersect [Oc] ED
point P : intersect mn [Ob]
point Q : intersect mn [Oc]
MP = NQ : ?
```


# 3. IMO 2019 SL G3 == IMO 2019 P2

In triangle ABC, point A1 lies on side BC and point B1 lies on side AC. Let P and Q be points on segments AA1 and BB1, respectively, such that PQ is parallel to AB. Let P1 be a point on line PB1, such that B1 lies strictly between P and P1, and ∠PP1C = ∠BAC. Similarly, let Q1 be a point on line QA1, such that A1 lies strictly between Q and Q1, and ∠CQ1Q = ∠CBA. Prove that points P, Q, P1, and Q1 are concyclic.

```java
normal triangle <ABC> :  // assumption: we can always construct a normal triangle
point A1 : between B C
point B1 : between A C
point P : between A A1
line pq : parallel P ab
point Q : intersect pq BB1
point P1 : copy CAB C P PB1  // copy angle CAB to legs C P and side PB1
point Q1 : copy CBA C Q QA1
circle [O] contains P1 P Q1 Q : ?
{1}
circle [O1] : contains A B C 
point A2 : intersect aa1 [O1]
point B2 : intersect bb1 [O1]
BAA2 = BB2A2 : contains [O1] A B C B2 A2
circle [C0] contains Q P B2 A2 : QPA2 = QB2A2
{2}
QPA2 = BAA2 : pq \\ ab  // pq is parallel to ab
CBA = CA2A1 : [O1] contains A B C A2
circle [C1] contains C Q1 A2 A1 : CQ1A1 = CA2A1
QQ1A2 = A1CA2 : [C1] contains C Q1 A2 A1 
A1CA2 = BAA2 : [O1] contains A B C A2
[C0] contains Q1 : QPA2 = QQ1A2
{3:2 P:Q, A:B, A1:B1, A2:B2, P1:Q1, P2:Q2} // repeat {2} with replacements
ABB2 = PQB2 : pq \\ ab
CAB = CB2B1 : [O1] contains A B C B2
circle [C2] contains C P1 B2 B1 : CP1B1 = CB2B1
PP1B2 = B1CB2 : [C2] contains C P1 B2 B1 
B1CB2 = ABB2 : [O1] contains A B C B2
[C0] contains P1 : PQB2 = PP1P2
```

# X1. IMO 2019 SL G4
Let P be a point inside triangle ABC. Let AP meet BC at A1, let BP meet CA at B1, and let CP meet AB at C1. Let A2 be the point such that A1 is the midpoint of PA2, let B2 be the point such that B1 is the midpoint of PB2, and let C2 be the point such that C1 is the midpoint of PC2. Prove that points A2, B2, and C2 cannot all lie strictly inside the circumcircle of triangle ABC.

```java
normal triangle <ABC> :
point P : inside <ABC>
point A1 : intersect ap BC
point B1 : intersect bp CA
point C1 : intersect cp AB
point A2 : mirror P A1
point B2 : mirror P B1
point C2 : mirror P C1
circle [O] : A B C
A2 inside [O] :
B2 inside [O] :
C2 not inside [O] : ?  // This problem cannot be solved by current formulation.
```

# X2. IMO 2019 SL G5

Let ABCDE be a convex pentagon with CD “ DE and =EDC ‰ 2  ̈ =ADB. Suppose that a point P is located in the interior of the pentagon such that AP “ AE and BP “ BC. Prove that P lies on the diagonal CE if and only if area(BCD) + area(ADE) = area(ABD) + area(ABP).

```java
convex pentagon {ABCDE} :
line l : copy ADB E D
line d : bisect EDC
distinct l d :
circle [A] : AE
circle [B] : BC
point P : intersect [A] [B] AB_hp[D]
area<BCD> + area<ADE> = area<ABD> + area<ABP> :
P between C E : ?
```

```java
convex pentagon {ABCDE} :
line l : copy ADB E D
line d : bisect EDC
distinct l d :
circle [A] : AE
circle [B] : BC
point P : intersect [A] [B] AB_hp[D]
P between C E :
area<BCD> + area<ADE> = area<ABD> + area<ABP> : ?
```

# 4. IMO 2019 SL G6

Let I be the incentre of acute-angled triangle ABC. Let the incircle meet BC, CA, and AB at D, E, and F, respectively. Let line EF intersect the circumcircle of the triangle at P and Q, such that F lies between E and P. Prove that DPA + AQD = QIP.

```java
acute triangle <ABC> :
circle [I], point I, D, E, F : incircle <ABC>, BC, CA, AB
cirlce [O] : A B C
point P : intersect ef [O] ca_hp[F]
point Q : intersect ef [O] ab_hp[E]
line l : copy DPA Q I
AQD = PIl : ?
```


# 5. IMO 2019 SL G7 == IMO 2019 P6

Let ABC be a triangle with incenter I and incircle ω. Let D, E, F denote the tangency points of ω with BC, CA, AB. The line through D perpendicular to EF meets ω again at R (other than D), and line AR meets ω again at P (other than R). Suppose the circumcircles of 4P CE and 4P BF meet again at Q (other than P). Prove that lines DI and P Q meet on the external ∠A-bisector.

```java
normal triangle <ABC> :
circle [I], point I, D, E, F : incircle <ABC>, BC, CA, AB
line dr : perp D ef
point R : intersect dr [I]
point P : intersect ar [I]
circle [Ce] : P C E
circle [Cf] : P B F
point Q : intersect [Ce] [Cf]
point T : intersect [di] [pq]
line at : outer BAC
at contains T : ?
```


# X3. IMO 2019 SL G8

Let L be the set of all lines in the plane and let f be a function that assigns to each line l \in L a point f(l) on l. Suppose that for any point X, and for any three lines l1, l2, l3 passing through X, the points f(l1), f(l2), f(l3) and X lie on a circle.

Prove that there is a unique point P such that f(l) = P for any line l passing through P.

# 6. IMO 2020 P1

Consider the convex quadrilateral ABCD. The point P is in the interior of ABCD. The following ratio equalities hold:
∠PAD : ∠PBA : ∠DPA = 1 : 2 : 3 = ∠CBP : ∠BAP : ∠BPC.
Prove that the following three lines meet in a point: the internal bisectors of angles ∠ADP and ∠PCB and the perpendicular bisector of segment AB.

```java
normal triangle {APB} :
line bc : copy 0.5 BAP P B
line pc : copy 1.5 BAP B P
point C : intersect bc pc
line ad : copy 0.5 ABP P A
line pd : copy 1.5 ABP A P
point D : intersect ad pd
line lc : bisector BCP
line ld : bisector ADP
point I : intersect lc ld
line lp : bisector AB
lp contains I : ?
{1}
circle [O] : A B P
OA = OP = OB : O center [O]
{2}
BOP / BAP = 2 : O center [O]
circle [I1] contains C P O B : BOP = BC!P  // both = 2 BAP
OPB = OBP : SAS {OPB} = {OBP}
OBP = OCP : [I1] contains O B C P
OPB = OCB : [I1] contains O B C P
co == lc : autoelim
{3:2 B:A C:D [I1]:[I2]}
  AOP / ABP = 2 : O center [O]
  circle [I2] contains D P O A : AOP = AD!P  // both = 2 ABP
  OPA = OAP : SAS {OPB} = {OBP}
  OPA = ODA : [I1] contains O A D P
  OAP = ODP : [I1] contains O A D P
  do == ld : autoelim
{4}
O == I : autoelim
lp contains I : IA = IB
```

# 7. IMO 2018 SL G1 == IMO 2018 P1

Let ABC be an aute-angled triangle with irumirle Γ. Let D and E be points on
the segments AB and AC , respetively, suh that AD = AE . The perpendiular bisetors of the segments BD and CE interset the small arcs AB and AC at points F and G respetively. Prove that DE // FG.

```java
acute triangle {ABC} :
point D : between A B
point E : copy AD A C
line lf : bisect B D
line lg : bisect E C
circle [O] : A B C
point F : intersect lf [O] ab_hp[!C]
point G : intersect lg [O] ca_hp[!B]
de \\ fg : ?
{1}
FDB = FBD : F on lg
point K : intersect fd [O]
FBD = AKD : [O] contains A F B K
AD = AK : ASA {ADK} = {AKD}
{2: D:E F:G B:C K:L lf:lg}
  GEC = GCE : G on lg
  point L : intersect ge [O]
  GCE = ALE : [O] contains A G C L
  AE = AL : ASA {AEL} = {ALE}
{3}
circle [A] contains L D E K : AL = AD = AE = AK
LED = LKD : [A] contains L D E K
LKD = LGF : [O] contains L K G F
de \\ fg : LED = KGF
```

# 8. IMO 2018 SL G2

Let ABC be a triangle with AB “ AC , and let M be the midpoint of BC . Let P be
a point suh that P B < P C and P A is parallel to BC . Let X and Y be points on the lines P B and P C , respetively, so that B lies on the segment P X , C lies on the segment P Y , and ^PXM = ^PYM . Prove that the quadrilateral APXY is cylic


```java
isosecles {ABC} : 
point M : midpoint B C
line ap : parallel A bc
point P : free ap am_hp[B]
point X : free bp bc_hp[!P]
point Y : copy BXM C M !P
circle [O] : A X Y
[O] contains P : ?
{1}
AMB = AMC = 90 : SSS {AMB} {AMC}
line yz : perp Y pc
point Z : intersect yz am
circle [I1] contains P A Y Z : PAZ = PYZ = 90
circle [I2] contains C Y M Z : CYZ = CMZ = 90
CYM = CZM : [I2] contains C Y M Z
CZM = BZM : SAS {BMZ} = {CMZ}
circle [I3] contains B M Z X : BXM = BZM
BXZ = CMZ = 90 : [I3] contains B M Z X
circle [I1] contains P Y X Z : PYZ = PXZ = 90
[O] == [I1] : A X Z
```

# X4. IMO 2018 SL G3

A cirle ω of radius 1 is given. A colletion T of triangles is called good, if the following conditions hold:
(i) each triangle from T is inscribed in ω;
(ii) no two triangles from T have a common interior point.
Determine all positive real numbers t such that, for each positive integer n, there exists a good colletion of n triangles, each of perimeter greater than t.

```java
```

# 9. IMO 2018 SL G4

A point T is chosen inside a triangle ABC . Let A1 , B1, and C1 be the refletions of T in BC , CA, and AB , respetively. Let Ω be the circumcirle of the triangle A1B1C1. The lines A1T , B1T , and C1T meet Ω again at A2 , B2 , and C2, respectively. Prove that the lines AA2 , BB2 , and CC2 are concurrent on Ω

```java
triangle {ABC} :
point T : free ab_hp[C] bc_hp[A] ca_hp[B]
point A1 : mirror T bc A3
point B1 : mirror T ca B3
point C1 : mirror T ab C3
circle [O] : A1 B1 C1
point A2 : intersect ta1 [O]
point B2 : intersect tb1 [O]
point C2 : intersect tc1 [O]
point X : intersect aa2 bb2
cc2 contains X : ?
{1}
point K : intersect [O] cc2
{a}
  {2}
  CT = CA1 : SAS {TA3C} = {A1A3C}
  CT = CB1 : SAS {TB3C} = {B1B3C}
  circle [C] contains T A1 B1 : CT = CA1 = CB1
  A1CB = TB1A1 : circle [C] contains T A1 B1
  TB1A1 = B2C2A1 : circle [O] contains A1 B1 B2 C2
  {3:2 B:C B1:C1 B2:C2 B3:C3}
    BT = BA1 : SAS {TA3B} = {A1A3B}
    BT = BC1 : SAS {TC3B} = {C1C3B}
    circle [B] contains T A1 C1 : BT = BA1 = BC
    A1BC = TC1A1 : circle [B] contains T A1 C1
    TC1A1 = C2B2A1 : circle [O] contains A1 C1 C2 B2
  {4}
  BA1B2 = CA1C2 : autoelim
  A1B2 / A1B = A1C2 / A1C : AA {A1BC} {A1B2C2}
  BB2A1 = CC2A1 : RAR {BB2A1} {CC2A1}
  b2k == bb2, bb2 contains K : autoelim
{b:a A:B A1:B1 A2:B2 A3:B3}
  // ...
  a2k == aa2, aa2 contains K : autoelim
K == X, cc2 contains X : autoelim
```

# 10. IMO 2018 SL G5

Let ABC be a triangle with circumcirle ω and incentre I . A line l intersects the
lines AI , BI , and CI at points D, E , and F , respectively, distinct from the points A, B, C, and I. The perpendicular bisectors x, y, and z of the segments AD, BE , and CF , respectively determine a triangle Θ. Show that the circumcirle of the triangle Θ is tangent to ω


```java
triangle {ABC} :
circle [O] : A B C
circle [I] : ab bc ca
point D : free ai
distinct D I A :
point E : free bi
distinct E B I :
point F : intersect de ci
distinct F C I :
line l : D E F
line yz : bisect A D A0
line xy : bisect C F C0
line zx : bisect B E B0
point X : intersect xy zx
point Y : intersect xy yz
point Z : intersect yz zx
circle [W] : X Y Z
point T, V : intersect [W] [O]
T == V : ?
{0}
point X0 : intersect ai [O]
point Y0 : intersect bi [O]
point Z0 : intersect ci [O]
x0y0 perp ci : known
y0z0 perp ai : known
z0x0 perp bi : known
X0Y0Z0 = ABC, Y0Z0X0 = BCA, Z0X0Y0 = CAB : autoelim 
X0Y0/XY=Y0Z0/YZ=Z0X0/ZX : AAA {X0Y0Z0} {XYZ}
{1a}
point Lx : intersect l x
line la : Lx A 
LxDA = LyAD : SAS {LxA0D} = {LxA0A}
{1b:1a A:B D:E}
  point Ly : intersect l y
  line lb : Ly B
  LyEB = LyBE : SAS {LyB0E} = {LyB0B}
{1c:1a A:C D:F}
  point Lz : intersect l z
  line lc : Lz C
  LzFC = LzCF : SAS {LzC0F} = {LzC0C}
{2}
point Tbc : intersect lb lc
BTbcC = BAC : autoelim  
// BTC = 180-2 ZXY = 180 - 2 Z0X0T0 
// = 2 BIC - 180 = 2 (180 - 0.5B - 0.5C) - 180
// = 180 - B - C = A
[O] contains A B C Tbc : BTbcC = BAC 
{3:2 lc:la A:C}
  point Tab : intersect lb la
  BTabA = BCA : autoelim
  [O] contains A B C Tab : BTabA = BCA
{4:2 lb:la B:A}
  point Tca : intersect lb lc
  ATcaC = ABC : autoelim
  [O] contains A B C Tca : ATcaC = ABC 
{4}
Tab == Tbc : intersect lb [O] = (Tab, Tbc, B) , Tab != B , Tbc != B
Tca == Tab : intersect la [O] = (Tca, Tab, A) , Tca != A , Tab != A
// let T = Tab = Tbc = Tca
{5}
line dy : perp D zx
point Db : intersect dy lb
point Yd : intersect zx dy
Db Ly Yd = D Ly Yd , LyDb = LyD : ASA {Ly Yd Db} = {Ly Yd D}
XDb = XD : SAS {Db Ly X} = {D Ly X}
{6:5 lb:lc B0:C0 Ly:Lz}
  line dz : perp D xy
  point Dc : intersect dz lc
  point Zd : intersect xy dz
  Dc Lz Zd = D Lz Zd , LzDc = LzD : ASA {Lz Zd Dc} = {Lz Zd D}
  XDc = XD : SAS {Dc Lz X} = {D Lz X}
{7}
DbXDc = pi - DbTDc : autoelim
// DbXDc = 2 ZXY = 180 - A = 180 - BTC = 180 -DbTDc
circle [C1] contains X Db T Dc : DbXDc = pi - DbTDc
DbTX = DcTX : circle [C1] contains X Db T Dc, XDb = XDc
DbTX = 0.5 DbTC = 0.5 BAC : autoelim
DbTX0 = BAX0 : [O] contains B A T X0
BAX0 = 0.5 BAC : autoelim
tx0 == tx : autoelim
{8:5}
{9:7}
tz0 == tz : autoelim
{10}
TX0/TX = Z0X0/ZX : AAA {TZ0X0} {TX0X0}
point T0 : intersect yy0 xx0
T0X0/T0X = X0Y0/XY : AAA {T0X0Y0} {T0XY}
T0 == T : autoelim
// TODO(thtrieu): think about this next.
```

# 11. IMO 2018 SL G6

A convex quadrilateral ABCD satisfies AB*CD = BC*DA. A point X is chosen inside the quadrilateral so that XAB = XCD and XBC = XDA . Prove that AXB + CXD = pi

```java
convex {ABCD} :
AB/BC = DA/DC :
point X : free ab_hp[C] bc_hp[D] cd_hp[A] da_hp[B]
XAB = XCD :
XBC = XDA :
AXB + CXD = pi : ?
```

# 12. IMO 2018 SL G7

```java
acute triangle {ABC} :
circle [O] : A B C
point P : free [O]
cirlce [Oa] : A O P
cirlce [Ob] : B O P
cirlce [Oc] : C O P
line la : perp bc Oa
line lb : perp ca Ob
line lc : perp ab Oc
point La : intersect lb lc
point Lb : intersect lc la
point Lc : intersect la lb
circle [I] : La Lb Lc
point X1, X2 : intersect [I] op
X1 == X2 : ?
{1}
line l : bisect O P
l contains Ob : ObO = ObP
l contains Oa : OaO = OaP
l contains Oc : OcO = OcP
{2}
 : SSS {OCP} = {OPC}
 : SSS {OcCO} = {OcPO}
OOc perp PC : autoelim
{3:2 C:A Oc:Oa}
  OOa perp PA : autoelim
{4:2 C:B Oc:Ob}
  OOb perp PB : autoelim
{5}
OaPOc = OaOOc : SSS {POaOc} = {OcOaO}
OaOOc = APC : autoelim
APC = ABC : [O] contains A B P C 
ABC = OcLbOa : autoelim
[Wb] contains Lb Oa Oc P : OcLbOa = OaPOc
{6:5 Lb:Lc Ob:Oc B:C}
  [Wc] contains Lc Ob Oa P
{7:5 Lb:La Oa:Ob A:B}
  [Wa] contains La Oc Ob P
{8}
LaLcP = OcOaP : [Wc] contains Lc Ob Oa P
OcOaP = LaLbP : [Wb] contains Lb Oa Oc P
[I] contains P : LaLcP = LaLbP
{9}
AOOa = POOa = pi - ACP : [O] contains A P C
PLcLa = POaOc : [Wc] contains Lc Ob Oa P
POaOc = OcOaO : SSS {POaOc} = {OcOaO}
OcOaO = APO : autoelim
APO = pi/2 - (AOP/2) = pi/2 - (pi - ACP) = ACP - pi/2 : autoelim
ACP - pi/2 = <LaLc, pc> : autoelim
dCP == dLcP : autoelim
pc contains Lc, APO = PLcLA : dCP == dLcP
{10}
OP tangent [I] : APO = PLcLA
```

# 13. IMO 2017 SL G1

Let ABCDE be a convex pentagon such that AB = BC = CD, EAB = BCD, and EDC = CBA. Prove that the perpendicular line from E to BC and the line segments AC and BD are concurrent.

```java
convex {ABCDE} :
AB = BC = CD :
EAB = BCD :
EDC = CBA :
line ei : perp E bc
point H : intersect bd ac
ei contains H : ?
{1}
line bi : perp B AC
line ci : perp C BD
point I : intersect bi ci
point T : intersect ih bc
ih perp bc : bh perp ic, ch perp ib
_ : SSS {BAC} = {BCA}
_ : SSS {CBD} = {CDB}
_ : SAS {IBA} = {IBC}
_ : SAS {ICB} = {ICD}
IAB = IAE : autoelim
IDE = IDC : autoelim
IEA = IED : AI bisect A, BI bisect B, CI bisect C, DI bisect D
ie perp bc : autoelim
ie == ih : perp bc
ei contains H
```

# 14. IMO 2017 SL G2

Let R and S be distinct points on circle O, and let t denote the tangent line to O at R. Point R' is the reflection of R w.r.t S. A point I is chosen on the smaller arc RS of O so that the circumcircle W of triangle ISR' intersect t at two different points. Denote by A the common point of W and t that is closest to R. Line AI meets O again at J. Show that JR' is tangent to W.

```java
circle [O] :
point R : free [O]
point P : mirror R O
point S : free [O]
line t : tangent [O] R
point R0 : mirror R S
point I : free [O] rs_hp[!P]
circle [W] : I S R0
point A, B : intersect [W] t
A != B : 
point J : intersect ai [O]
jr0 tangent [O] : ?
{1}
JRS = JIS : [0] contains J R I S
JIS = AR0S : [W] contains A I S R0
SJR = SRA : ra tangent [O]
R0R/RJ = AR0/SR : AAA {ARR0} ~ {SJR}
// Rearrange: R0R/AR0 = RJ/SR0 because SR = SR0
RR0J = R0AS : RAR {RR0J} ~ {R0AS}
jr0 tangent [W] : RR0J = R0AS
```

Solution 2:

```java
JRS = JIS : [0] contains J R I S
JIS = AR0S : [W] contains A I S R0
RJ \\ AR0 : JRS = AR0S
A0 : mirror A S
SRA0 = SR0A : SAS {RSA0} = {R0SA}
ra0 == rj : autoelim
SR0A0 = SRA : SAS {R0SA0} = {RSA}
SRA = SJR : ra tangent [O]
cirlce [T] contains J S A0 R0 : SR0A0 = SJR
SR0J = SA0J : cirlce [T] contains J S A0 R0
SA0J = SAR0 : autoelim
jr0 tangent [W]: SR0J = SAR0
```

# 15. IMO 2017 SL G3

Let O be the circumcenter of an acute scalene triangle ABC. Line OA intersects the altitudes of ABC through B and C at P and Q respectively. The altitudes meet at H. Prove that the circumcenter of triangle PQH lies on a median of triangle ABC.

```java
normal triangle ABC :
circle [O] : A B C
line bh : perp B ca
line ch : perp C ab
point H : intersect bh ch
point P : intersect oa bh
point Q : intersect oa ch
circle [T] : O P H
point N : midpoint B C
an contains T : ?
{1}
PQH = pi/2 - OAB : autoelim
OAB = pi/2 - AOB/2 : autoelim
AOB/2 = ACB : [O] contains A B C
// PQH = ACB
{2:1 B:C P:Q}
  PHQ = ABC
{3}
perp ah bc : perp bh ca, perp ch ab
AHP = ACB = PQH : autoelim
ah tangent [T] : AHP = PQH
point M : intersect at bc
line as : tangent [O] A
point S : intersect as BC
OSM = OAT : ?
circle [X] : O A S M
OAS = pi/2 : tangent as [O]
OMS = pi - OAS : cirlce [X] contains O A S M
MB = MC : rSAS {OBM} = {OCM}
N == M, an == am, an contains T : autoelim
```


# 16. IMO 2017 SL G4

In triangle ABC, let W be the excircle opposite A. Let D, E, F be the points where W is tangent to lines BC, CA, AB respectively. The circle AEF intersects BC at P and Q. Let M be the midpoint of AD. Prove that the circle MPQ is tangent to W.

```java
normal triangle ABC :
circle [A_] D E F : bc ca ab bc_hp[!A]
circle [O] : A E F
point P, Q : intersect bc [O]
point M : midpoint A D
circle [X] : M P Q
tangent [X] [A_] : ?
{1}
point T : intersect ad [A_]
perp ae a_e : tangent ae [A_]
perp af a_f : tangent af [A_]
contains [O] A_ : AFA_ = AEA_ = pi/2
point N : midpoint D T
A_ND = A_NT = pi/2 : SSS {A_DN} = {A_TN}
contains [O] N : A_NA = A_EA = pi/2
DP/DA = DN/DQ : [O] contains A P Q N
DP/DM = DT/DQ : autoelim
contains [X] T : DP/DM = DT/DQ
line tr : tangent [X] T
point R : intersect tr bc
DRA_ = TRA_ : rSSS {DRA_} = {TRA_}
point N_ : intersect dt ra_
N_D = N_T : SAS {DRN_} = {TRN_}
N_ == N : autoelim
DNR = TNR = pi/2 : autoelim
RT^2 = RD^2 : autoelim
RD^2 = RN * RA_ : AAA {DRN} {A_RD}
RN * RA_ = RQ * RP : AAA {RQN} {RA_P}
point T_ : intersect rt [X]
RT * RT_ = RQ * RP : AAA {RQT} {RT_P}
RT_ = RT : autoelim
T_ == T : autoelim
tangent [X] [A_] :
```

# 17. IMO 2017 SL G5

Let ABCC1B1A1 be a convex hexagon such that AB = BC , and suppose that the line
segments AA1 , BB1, and CC1 have the same perpendiular bisetor. Let the diagonals AC1 and A1C meet at D, and denote by ω the circle ABC . Let ω intersect the circle A1BC1 again at E != B . Prove that the lines BB1 and DE intersect on ω

```java
triangle {ABC}, AB=BC :
line l : free ???
point A1 : mirror A l Ma
point B1 : mirror B l Mb 
point C1 : mirror C l Mc
point D : intersect ac1 a1c
circle [W] : A B C
cirlce [O] : A1 B C1
point E, B : intersect [W] [O]
E != B :
point X : intersect bb1 de
[W] contains X : ?
{0}
point M : mirror B W
[W] contains M : WB = WM 
BAM = pi/2 : diameter_angle [W] B M A
BCM = pi/2 : diameter_angle [W] B M C
MA = MC : rSSS {BAM} {BCM}
{1}
circle [T] : A A1 C C1
point R : radical_center [W] [O] [T]
contains l R : ...
DA = DA1 : bisector l AA1 D
RA = RA1 : bisector l AA1 R
ADR = A1DR : SSS {RAD} {RA1D}
BEC = REA : midarc B AC
line ds : bisect ADC
point S : intersect ds ac
point S_ : intersect em ac
AEM = CEM : MA = MC
AS/CS = AD/CD : bisect ds ADC
AD/CD = AR/CR : bisect dr ADC
AR/CR = AE/CE : bisect er CEA
AE/CE = AS_/CS_ : bisect es_ CEA
S_ == S : autoelim
RDS = RES = pi/2 : autoelim
contains [P] R E S D : RES = RDS = pi/2
{3}
point X : intersect bb1 de
EXB = EDS : autoelim
EDS = ERS : [P] contains R E S D
ERS = EMB : autoelim
[W] contains [X] : EXB = EMB
```


# X5. IMO 2017 SL G6

Let n>=3 be an integer. Two regular n-gons A and B are given in the plane. Prove that the vertices of A that lie inside B or on its boundary are consecutive.

# 18. IMO 2017 SL G7

Convex ABCD has inscribed circle center I. Let Ia, Ib, Ic and Id be the incenters of triangles DAB, ABC, BCD, CDA respectively. Suppose that the common external tangents of the circles AIbId and CIbId meet at X, and the common external tangents of circles BIaIc and DIaIc meet at Y. Prove that XIY = pi/2

```java
convex {ABCD}, circle [I] ab bc cd da :
circle [Ia] da ab bd :
circle [Ib] ab bc ca :
circle [Ic] bc cd db :
circle [Id] cd da ac :
circle [Oa] A Ib Id :
circle [Ob] B Ia Ic :
circle [Oc] C Ib Id :
circle [Od] D Ia Ic :
line l1, l2, Ta, Tc : common_external_tangent [Oa] [Oc]
point X : intersect l1 l4
line l3, l4, Tb, Td : common_external_tangent [Ob] [Od]
point Y : intersect l3 l4
XIY = pi/2 : ?
{1}
point T : intersect [Ib] ca
point T_ : intersect [Id] ca
AB - BC = AD - CD : incircle I AB BC CD DA
AT = (AB+AC-BC)/2 : incircle Ib AB BC CA
AT_ = (AD+AC-CD)/2 : incircle Id AD AC CD
AT = AT_ : autoelim
T == T_ : autoelim
line ld : Id T
perp ld ca : tangent Id ca T
line lb : Ib T
perp lb ca : tangent Ib ca T
ld == lb : autoelim
{2:1 D:A B:C Id:Ia Ib:Ic}
  perp IaIc bd
{3}
line ld : bisect ADC
line lb : bisect ABC
contain ld I : tangent [I] da dc
contain lb I : tangent [I] ab bc
contain ld Id : tangent [Id] da bc
contain lb Ib : tangent [Ib] ab bc
OaAId = TAIb : contains [Oa] A Ib Id, perp AT IbId
IdAD = IdAC : tangent [I] da ac
OaAD = OaAB = DAB/2 : autoelim
l_OaA == l_AI : autoelim
contain l_AI Oa : autoelim
{4:3 Oa:Ob D:A B:C}
  contain l_BI Ob : autoelim
{5:3 Oa:Oc D:B B:D}
  contain l_CI Oc : autoelim
{6:3 Oa:Od D:C B:A}
  contain l_DI Od : autoelim
{7}
perp l_OaOc Id Ib : OaId=OaIb, OcId=OcIb
point W : intersect l_OaOc l_IbId
point Ta : intersect l1
OaX/OcX = OaTa/OcTc : rAAA TaXOa TcXOc
OaTa/OcTc = OaIb/OcIb = 
```







