Below is an example playing with interactive sketch exploration that builds the Thales theorem premise. Please ignore all the `# comment`.

```bash
Depth = 0
bc ab ca
0. ConstructMidPoint
1. ConstructMirrorPoint
2. ConstructIntersectSegmentLine
3. ConstructParallelLine
4. ConstructThirdLine
5. EqualAnglesBecauseParallel
6. ParallelBecauseCorrespondingAngles
7. ParallelBecauseInteriorAngles
8. SAS
9. ASA
[Q]uit [I]nspect [P]db >> 0  # Construct mid point
A B = A B  # of segment AB
   B::B  A::A  l::ab   (0.000937s)  # detected a match in 0.000937s
[Y]es [N]ext [E]scape [I]nspect [P]db >> y  # choose this match
Build Point P1 such that  # the mid point is named P1
   * ab[P1]
   * s1
   * P1[s1
   * A[s1
   * s2
   * P1[s2
   * B[s2
   * 1m_0
   * s1=1m_0
   * s2=1m_0
 * copy 0.00168013572693  # time copy canvas and state
 * add rel 0.000244140625  # time to update state
 * draw 0.00111603736877  # time to draw on canvas
 * add spatial rel 0.000128030776978  # time to add topological inspections into state

Depth = 1
bc ab ca
0. ConstructMidPoint
1. ConstructMirrorPoint
2. ConstructIntersectSegmentLine
3. ConstructParallelLine
4. ConstructThirdLine
5. EqualAnglesBecauseParallel
6. ParallelBecauseCorrespondingAngles
7. ParallelBecauseInteriorAngles
8. SAS
9. ASA
[Q]uit [I]nspect [P]db >> 3  # Construct parallel line
A l = P1 bc  # through P1 and parallel to bc
   A_2::P1  l_2::bc   (0.000551s)
[Y]es [N]ext [E]scape [I]nspect [P]db >> y
Build LineDirection d1_0 such that
   * bc|d1_0
Build Line l1_0 such that  # new line is named l1_0
   * l1_0[P1]
   * l1_0|d1_0
 * copy 0.00137805938721
 * add rel 0.000160932540894
 * draw 0.00135898590088
 * add spatial rel 0.000201940536499

Depth = 2
ab ca bc l1_0
0. ConstructMidPoint
1. ConstructMirrorPoint
2. ConstructIntersectSegmentLine
3. ConstructParallelLine
4. ConstructThirdLine
5. EqualAnglesBecauseParallel
6. ParallelBecauseCorrespondingAngles
7. ParallelBecauseInteriorAngles
8. SAS
9. ASA
[Q]uit [I]nspect [P]db >> 2  # Construct intersection of line and segment
A B l = A C l1_0  # segment AC and line l1_0
   A_1::A  B_1::C  ab::ca  l_1::l1_0   (0.000568s)
[Y]es [N]ext [E]scape [I]nspect [P]db >> y
Build Point P2 such that
   * ca[P2]
   * l1_0[P2]
 * copy 0.00146293640137
 * add rel 0.000128984451294
 * draw 0.00576996803284
 * add spatial rel 6.103515625e-05
 ```