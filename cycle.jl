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

n = 28		# exact n=28 energy is -12.2254405486
#  Make initial product state in up down up down up down pattern (Neel state)
# Make first tensor a 1 x 2 x m tensor; and last is m x 2 x 1  (rather than vectors)
A = [zeros(1,2,1) for i=1:n]
for i=1:n
  A[i][1,iseven(i) ? 2 : 1,1] = 1.0
end

function mainLoop()
  numIters = [1000,2000,8000]
  #numIters = [100,200,1000]
  taus = [.1,.01,.001]
  for stage = 1:3
    tau = taus[stage]
    numIter = numIters[stage]
    for iter = 1:numIter
      taugate = reshape(expm(-tau * reshape(Htwosite,4,4)),2,2,2,2)
      #println("\n iteration = $iter")
      for j = 1:N
        applyGateAndTrim(j,taugate)
      end
    end
    println("\n End of stage $stage")
  end
end

function normEnv(j)
  nHalf = Int64(ceil(n/2))
  nHalfm1 = mod(nHalf-2,n)+1
  jm1 = mod(j-2,n)+1
  jp1 = mod(j,n)+1
  jp2 = mod(j+1,n)+1
  mid = mod(j+nHalf,n)+1
  Eleft = calcEnv(nHalf,jm1,true)
  Eright = calcEnv(jp2,nHalfm1,false)
end

function calcEnv(l,r,toRight)
  ld = size(A[l],1)
  E = eye(ld)
  num = r >= l? l-r+1: n-l+r+1
  curr = l
  for k = 1:num
    Nc = N[curr]
    Ncp = Nc'
    if (toRight)
      @tensor begin
        Enew[c,d] := E[a,b]*Ncp[a,p,c]*Nc[b,p,d]
      end
      curr = mod(curr,n)+1
    else
      @tensor begin
        Enew[a,b] := E[c,d]*Ncp[a,p,c]*Nc[b,p,d]
      end
      curr = mod(curr-2,n)+1
    end
    E = Enew
  end
  return(E)
end
