function dosvdtrunc(AA,m)		# AA a matrix;  keep at most m states
  (u,d,v) = svd(AA)
  prob = dot(d,d)		# total probability
  mm = min(m,length(d))	# number of states to keep
  d = d[1:mm]			# middle matrix in vector form
  trunc = prob - dot(d,d)
  U = u[:,1:mm]
  V = v[:,1:mm]'
  (U,d,V,trunc)		# AA == U * diagm(d) * V	with error trunc
end

function dosvdleftright(AA,m,toright)
  (U,d,V,trunc) = dosvdtrunc(AA,m)
  if toright
    V = diagm(d) * V
  else
    U = U * diagm(d)
  end
  (U,V,trunc)
end

function dosvd4(AA,m,toright)	# AA is ia * 2 * 2 * ib;  svd down the middle;  return two parts
  ia = size(AA,1)
  ib = size(AA,4)
  AA = reshape(AA,ia*2,2*ib)
  (U,V,trunc) = dosvdleftright(AA,m,toright)
  mm = size(U,2)
  U = reshape(U,ia,2,mm)
  V = reshape(V,mm,2,ib)
  (U,V,trunc)
end

using TensorOperations

function JK(a,b)	# Julia kron,  ordered for julia arrays; returns matrix
  (a1,a2) = size(a)
  (b1,b2) = size(b)
  reshape(Float64[a[i,ip] * b[j,jp] for i=1:a1, j=1:b1, ip=1:a2, jp=1:b2],a1*b1,a2*b2)
end

function JK4(a,b)	# Julia kron,  ordered for julia arrays, return expanded into 4 indices
  (a1,a2) = size(a)
  (b1,b2) = size(b)
  Float64[a[i,ip] * b[j,jp] for i=1:a1, j=1:b1, ip=1:a2, jp=1:b2]
end

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

function test()
    tau = .1
    taugate = reshape(expm(-tau * reshape(Htwosite,4,4)),2,2,2,2)
    for j = 1:n
        normEnv(j)
        @show(j)
        @show(calcOverlapCycle(A,A))
        jp1 = mod(j,n)+1
        (A[j],A[jp1]) = applyGateAndTrim(A[j],A[jp1],taugate)
        @show(calcOverlapCycle(A,A))
        println(" ")
    end
end

function mainLoop()
  #numIters = [1000,2000,8000]
  numIters = [100,200,1000]
  taus = [.1,.01,.001]
  @show(calcOverlapCycle(A,A))
  for stage = 1:3
    tau = taus[stage]
    numIter = numIters[stage]
    for iter = 1:numIter
      taugate = reshape(expm(-tau * reshape(Htwosite,4,4)),2,2,2,2)
      #println("\n iteration = $iter")
      for j = 1:n
        normEnv(j)
        jp1 = mod(j,n)+1
        (A[j],A[jp1]) = applyGateAndTrim(A[j],A[jp1],taugate)
      end
    end
    println("\n End of stage $stage")
    @show(calcOverlapCycle(A,A))
  end
end

function normEnv(j)

  nHalf = Int64(ceil(n/2))
  jm1 = mod(j-2,n)+1
  jp1 = mod(j,n)+1
  jp2 = mod(j+1,n)+1
  mid = mod(j+nHalf,n)+1
  midp1 = mod(mid,n)+1
  Eleft = calcEnv(midp1,jm1,true)
  Eright = calcEnv(jp2,mid,false)
  Eleft = 0.5*(Eleft+Eleft')
  Eright = 0.5*(Eright+Eright')

  F = eigfact(Eleft)
  d = F[:values]
  U = F[:vectors]
  (d,U) = cleanEigs(d,U)
  dSqrt = sqrt.(d)
  s = size(A[jm1])
  Ajm1 = reshape(A[jm1],s[1]*s[2],s[3])
  #@show(size(A[jm1]),size(U),size(dSqrt))
  Ajm1 = Ajm1*U*diagm(inv.(dSqrt))
  A[jm1] = reshape(Ajm1,s[1],s[2],length(d))
  s = size(A[j])
  Aj = reshape(A[j],s[1],s[2]*s[3])
  Aj = diagm(dSqrt)*U'*Aj
  A[j] = reshape(Aj,length(d),s[2],s[3])

  F = eigfact(Eright)
  d = F[:values]
  U = F[:vectors]
  (d,U) = cleanEigs(d,U)
  dSqrt = sqrt.(d)
  s = size(A[jp2])
  Ajp2 = reshape(A[jp2],s[1],s[2]*s[3])
  Ajp2 = diagm(inv.(dSqrt))*U'*Ajp2
  A[jp2] = reshape(Ajp2,length(d),s[2],s[3])
  s = size(A[jp1])
  Ajp1 = reshape(A[jp1],s[1]*s[2],s[3])
  Ajp1 = Ajp1*U*diagm(dSqrt)
  A[jp1] = reshape(Ajp1,s[1],s[2],length(d))

end

function calcEnv(l,r,toRight)
  ld = toRight? size(A[l],1): size(A[r],3)
  E = eye(ld)
  num = r >= l? r-l+1: n-l+r+1
  curr = toRight? l: r
  for k = 1:num
    Ac = A[curr]
    Acp = conj.(Ac)
    if (toRight)
      @tensor begin
        Enew[c,d] := E[a,b]*Acp[a,p,c]*Ac[b,p,d]
      end
      curr = mod(curr,n)+1
    else
      @tensor begin
        Enew[a,b] := E[c,d]*Acp[a,p,c]*Ac[b,p,d]
      end
      curr = mod(curr-2,n)+1
    end
    E = Enew
  end
  return(E)
end

function applyGateAndTrim(Aleft,Aright,g)
    @tensor begin
        ABg[a,s1p,s2p,c] := Aleft[a,s1,b]*Aright[b,s2,c]*g[s1,s2,s1p,s2p]
    end
    ABg = renormL2(ABg)
    a = size(ABg)
    ABg = reshape(ABg,a[1]*a[2],a[3]*a[4])
    (U,d,V) = svd(ABg)
    newDim = min(D,length(d))
    @show(length(d),newDim)
    U = U[:,1:newDim]
    V = V[:,1:newDim]
    diagD = diagm(d[1:newDim])
    @show(trace(ABg'*U*diagD*V'))
    @show(d[1:newDim])
    A2p = reshape(U,a[1],a[2],newDim)
    B2p = reshape(diagD*V',newDim,a[3],a[4])
    return(A2p, B2p)
end

function renormL2(T)
  t = size(T)
  Tvec = reshape(T,prod(t))
  norm = abs(
  Tvec'*Tvec)
  T = T/sqrt(norm)
  return(T)
end

function cleanEigs(d,U)
    count = 1
    while (d[count] <= 0)
        count += 1
    end
    newD = d[count:length(d)]
    newU = U[:,count:length(d)]
    return(newD,newU)
end

function calcOverlapCycle(T,S)

  left = eye(size(T[1])[1]*size(S[1])[1])
  for i = 1:n
    Siconj = conj.(S[i])
    Ti = T[i]
    @tensor begin
      NewM[u,w,x,y] := Siconj[u,s,x]*Ti[w,s,y]
    end
    nm = size(NewM)
    left = left*reshape(NewM,nm[1]*nm[2],nm[3]*nm[4])
  end
  norm = trace(left)
  return(norm)

end
