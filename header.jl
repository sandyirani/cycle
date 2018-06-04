



sz = Float64[0.5 0; 0 -0.5]
sp = Float64[0 1; 0 0]
sm = sp'
Htwosite = reshape(JK(sz,sz) + 0.5 * JK(sp,sm) + 0.5 * JK(sm,sp),2,2,2,2)
# order for Htwosite is s1, s2, s1p, s2p

D = 10
n = 28		# exact n=28 energy is -12.2254405486
#  Make initial product state in up down up down up down pattern (Neel state)
# Make first tensor a 1 x 2 x m tensor; and last is m x 2 x 1  (rather than vectors)
A = [zeros(1,2,1) for i=1:n]
for i=1:n
  A[i][1,iseven(i) ? 2 : 1,1] = 1.0
end
