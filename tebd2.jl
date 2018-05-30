include("utilities.jl")
include("header.jl")

function mainLoopLine()
  numIters = [1000,2000,8000]
  #numIters = [100,200,1000]
  taus = [.1,.01,.001]
  @show(calcOverlapCycle(A,A))
  for stage = 1:3
    tau = taus[stage]
    numIter = numIters[stage]
    taugate = reshape(expm(-tau * reshape(Htwosite,4,4)),2,2,2,2)
    for iter = 1:numIter
      #println("\n iteration = $iter")
      for j = 1:n-1
        normEnvLine(j)
        (A[j],A[j+1]) = applyGateAndTrim(A[j],A[j+1],taugate)
      end
    end
    println("\n End of stage $stage")
    @show(calcOverlapCycle(A,A))
    @show(calcEnergyLine())
  end
end

function normEnvLine(j)

  if (j > 1)
    Eleft = calcEnv(1,j-1,true)
    Eleft = 0.5*(Eleft+Eleft')
    F = eigfact(Eleft)
    d = F[:values]
    U = F[:vectors]
    (d,U) = cleanEigs(d,U)
    dSqrt = sqrt.(d)
    s = size(A[j-1])
    Ajm1 = reshape(A[j-1],s[1]*s[2],s[3])
    Ajm1 = Ajm1*U*diagm(inv.(dSqrt))
    A[j-1] = reshape(Ajm1,s[1],s[2],length(d))
    s = size(A[j])
    Aj = reshape(A[j],s[1],s[2]*s[3])
    Aj = diagm(dSqrt)*U'*Aj
    A[j] = reshape(Aj,length(d),s[2],s[3])
  end

  if (j < n-1)
    Eright = calcEnv(j+2,n,false)
    Eright = 0.5*(Eright+Eright')
    F = eigfact(Eright)
    d = F[:values]
    U = F[:vectors]
    (d,U) = cleanEigs(d,U)
    dSqrt = sqrt.(d)
    s = size(A[j+2])
    Ajp2 = reshape(A[j+2],s[1],s[2]*s[3])
    Ajp2 = diagm(inv.(dSqrt))*U'*Ajp2
    A[j+2] = reshape(Ajp2,length(d),s[2],s[3])
    s = size(A[j+1])
    Ajp1 = reshape(A[j+1],s[1]*s[2],s[3])
    Ajp1 = Ajp1*U*diagm(dSqrt)
    A[j+1] = reshape(Ajp1,s[1],s[2],length(d))
  end

end

function calcEnergyLine()
    norm = calcOverlapCycle(A,A)
    AE = [copy(A[j]) for j = 1:n]
    energy = 0
    for j = 1:n-1
        (AE[j],AE[j+1]) = applyGate(AE[j],AE[j+1],Htwosite)
        twoSiteE = calcOverlapCycle(AE,A)
        @show(twoSiteE)
        energy += twoSiteE
        AE[j] = copy(A[j])
        AE[j+1] = copy(A[j+1])
    end
    return(energy/(n*norm))
end


function test(k)
  Eleft = calcEnv(1,k-1,true)
  Eright = calcEnv(k+2,n,false)
  @show(Eleft)
  @show(Eright)
end
