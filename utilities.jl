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

function applyGateAndTrim(Aleft,Aright,g)
    @tensor begin
        ABg[a,s1p,s2p,c] := Aleft[a,s1,b]*Aright[b,s2,c]*g[s1,s2,s1p,s2p]
    end
    ABg = renormL2(ABg)
    a = size(ABg)
    ABg = reshape(ABg,a[1]*a[2],a[3]*a[4])
    (U,d,V) = svd(ABg)
    newDim = min(D,length(d))
    U = U[:,1:newDim]
    V = V[:,1:newDim]
    diagD = diagm(d[1:newDim])
    A2p = reshape(U,a[1],a[2],newDim)
    B2p = reshape(diagD*V',newDim,a[3],a[4])
    return(A2p, B2p)
end

function applyGate(Aleft,Aright,g)
    @tensor begin
        ABg[a,s1p,s2p,c] := Aleft[a,s1,b]*Aright[b,s2,c]*g[s1,s2,s1p,s2p]
    end
    ABg = renormL2(ABg)
    a = size(ABg)
    ABg = reshape(ABg,a[1]*a[2],a[3]*a[4])
    (U,d,V) = svd(ABg)
    U = U[:,1:length(d)]
    V = V[:,1:length(d)]
    diagD = diagm(d[1:length(d)])
    A2p = reshape(U,a[1],a[2],length(d))
    B2p = reshape(diagD*V',length(d),a[3],a[4])
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

function calcEnv(l,r,toRight)
  ld = toRight? size(A[l],1): size(A[r],3)
  E = eye(ld)/sqrt(ld)
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
