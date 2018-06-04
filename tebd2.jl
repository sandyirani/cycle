using TensorOperations
include("utilities.jl")
include("header.jl")

function mainLoopLine2()
  @show(calcOverlapCycle(A,A))
  tau = 0.2
  for swp = 1:10000
    swp%100 == 0 && (tau = 0.2*100/swp)
    taugate = reshape(expm(-tau * reshape(Htwosite,4,4)),2,2,2,2)
    #println("\n iteration = $iter")
    for j = 1:n-1
      normEnvLine(j)
      (A[j],A[j+1]) = applyGateAndTrim(A[j],A[j+1],taugate)
    end
    if swp%100 == 0
      @show(calcE())
      @show(calcEnergyLine())
      @show(calcOverlapCycle(A,A))
    end
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
        energy += (twoSiteE/norm)
        AE[j] = copy(A[j])
        AE[j+1] = copy(A[j+1])
    end
    return(energy)
end

function calcE()
  totE = 0
  AE = [copy(A[j]) for j = 1:n]
  for ii=-n+1:n-1		# if negative, going right to left
    ii == 0 && continue
    i = abs(ii)
    toright = ii > 0
    Ai = AE[i]
    Ai1 = AE[i+1]
    @tensor begin
      AA[a,b,d,e] := Ai[a,b,c] * Ai1[c,d,e]
      nor = scalar(AA[a,f,g,e] * AA[a,f,g,e])
      SdotS = scalar(AA[a,f,g,e] * Htwosite[f,g,fp,gp] * AA[a,fp,gp,e])
    end
    toright && (totE += SdotS/nor)
    AA *= 1.0 / sqrt(nor)
    (AE[i],AE[i+1],trunc) = dosvd4(AA,D,toright)
  end
  return(totE)
end


function test(k)
  Eleft = calcEnv(1,k-1,true)
  Eright = calcEnv(k+2,n,false)
  @show(Eleft)
  @show(Eright)
end
