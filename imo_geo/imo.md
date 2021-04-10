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
OcO = OcP : SSS {OcOP} = {OcPO}
OaOOc = OaPOc : SAS {OaOcO} = {OaOcP}
OaOOc = APC : autoelim
ABC = OcLbOa : autoelim
APC = ABC : [O] contains A B P C 
[Wb] contains Lb Oa Oc P : OaLbOc = OaPOc
{3:2 Lb:Lc Oa:Ob Oc:Oa P}
  [Wc] contains Lc Ob Oa P
{4:2 Lb:La Oa:Oc Oc:Ob P}
  [Wa] contains La Oc Ob P
{5}
LaLcP = OcOaP : [Wc] contains Lc Ob Oa P
OcOaP = LaLbP : [Wb] contains Lb Oa Oc P
[I] contains P : LaLcP = LaLbP
{6}

```











