include("utilities.jl")
include("header.jl")


function mainLoop()
  numIters = [1000,2000,8000]
  #numIters = [100,200,1000]
  taus = [.1,.01,.001]
  @show(calcOverlapCycle(A,A))
  for stage = 1:3
    tau = taus[stage]
    numIter = numIters[stage]
    for iter = 1:numIter
      taugate = reshape(expm(-tau * reshape(Htwosite,4,4)),2,2,2,2)
      #println("\n iteration = $iter")
      for j = 1:n
        #normEnv(j)
        jp1 = mod(j,n)+1
        (A[j],A[jp1]) = applyGateAndTrim(A[j],A[jp1],taugate)
      end
    end
    println("\n End of stage $stage")
    @show(calcOverlapCycle(A,A))
  end
  @show(calcEnergy())
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






function calcEnvFull(l,r)
    ld = size(A[l],1)
    E = eye(ld*ld)
    num = r >= l? r-l+1: n-l+r+1
    curr = l
    for k = 1:num
        Ac = A[curr]
        Acp = conj.(Ac)
        @tensor begin
            Enew[a,b,c,d] := Acp[a,p,c]*Ac[b,p,d]
        end
        curr = mod(curr,n)+1
        s = size(Enew)
        E = E * reshape(Enew,s[1]*s[2],s[3]*s[4])
    end
    return(E)
end

function calcEnergy()
    norm = calcOverlapCycle(A,A)
    AE = [copy(A[j]) for j = 1:n]
    energy = 0
    for j = 1:n
        jp1 = mod(j,n)+1
        (AE[j],AE[jp1]) = applyGate(AE[j],AE[jp1],Htwosite)
        energy += calcOverlapCycle(AE,A)
        AE[j] = copy(A[j])
        AE[jp1] = copy(A[jp1])
    end
    return(energy/(n*norm))
end
