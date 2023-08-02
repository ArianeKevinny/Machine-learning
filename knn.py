def normalization(dados):
    ## Normalização uilizando o Min-max
    for i in range(len(dados)):
        minimo = min(dados[i])
        maximo = max(dados[i])

        for j in range(len(dados[i])):
            aux = (dados[i][j] - minimo)/(maximo - minimo)
            dados[i][j] = aux
    
    ## Normalização utilizando Soma
    """
    for i in dados:
        soma = sum(dados[i])
        for j in range(len(dados[i])):
            dados[i][j] = dados[i][j]/soma    
    """

    return dados

def euclediana(x1, y1, z1, x2, y2, z2):
    aux = (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2
    return aux ** 0.5


def knn(dadoFebre, dadoEnjoo, dadoMancha, k):
#def knn():
    febre = [0, 1, 2, 2, 0]
    enjoo = [1, 0, 1, 0, 0]
    mancha = [3, 2, 3, 0, 4]
    diagnostico = ["DOENTE", "SAUDÁVEL", "DOENTE", "SAUDÁVEL", "DOENTE"]
    
    dados_normalizados = normalization([febre, enjoo, mancha])

    distancia = [float('inf')]*k
    knnLista = [0]*k

    for i in range(len(dados_normalizados[0])):
        maximo = max(distancia)
        dist = euclediana(
            dadoFebre, dadoEnjoo, dadoMancha, 
            dados_normalizados[0][i],
            dados_normalizados[1][i],
            dados_normalizados[2][i])
        if dist < maximo:
            j= distancia.index(maximo)
            distancia[j] = dist
            knnLista[j] = diagnostico[i]
    
    ## Verificando a classe de seus k vizinhos mais próximos
    qtdDoentes = knnLista.count('DOENTE')
    qtdSaudaveis = knnLista.count('SAUDÁVEL')

    febre.append(dadoFebre)
    enjoo.append(dadoEnjoo)
    mancha.append(dadoMancha)
    if qtdDoentes < qtdSaudaveis: 
        diagnostico.append("SAUDÁVEL")
    elif qtdDoentes > qtdSaudaveis:
        diagnostico.append("DOENTE")
    else:
        diagnostico.append(
            knnLista[distancia.index(min(distancia))])


    dados = {'febre': febre, 'enjoo': enjoo, 'mancha': mancha, 'diagnostico': diagnostico}
    print(dados)
    

knn(1, 0, 1, 2)