# Modelo de identificação de doenças agrícolas em plantas e frutos

## Resumo. 
Este Projeto tem como objetivo apresentar uma solução para os responsáveis na área de cultivo, lavoura ou agricultura na identificação de doenças agrícolas utilizando um modelo de Deep Learning capaz de, através da captura de imagens, verificar a saúde de qualquer espécie vegetal. A proposta tem como maior objetivo auxiliar no combate a pragas, insetos e doenças encontradas no setor, evitando a sua proliferação que consequentemente coloca em risco a produtividade e expõe à possibilidade de contaminação e destruição do Solo tal qual a saúde física dos consumidores e a saúde financeira dos agricultores. 
Palavras-chaves: Doenças Agrícolas, Modelo, Processamento de imagem, Deep Learning.

## 1.	Problema
Existem inúmeras patologias que afetam as plantações, tanto como suas folhas e frutos. De acordo com a Empresa Brasileira de Pesquisa Agropecuária (Embrapa) já foram registrados mais de 120 tipos diferentes de enfermidades Agrícolas apenas no Brasil. As desordens mapeadas pela Empresa classificam-se em Fungos, Vírus, Pragas, Bactérias, Fitotoxidez, Algas, Deficiências Nutricionais e Senescência.
Cada desordem apresenta um determinado tipo de sintoma nas folhas de Plata e nos frutos cultivados, de forma que é necessário alto conhecimento dos empreendedores do campo para saber o tipo de enfermidade, afim de ter o correto diagnostico e saber qual o devido tratamento deve dar para não por fim aos hectares plantados.
As desordens atingem em grande parte as principais culturas de interesse comercial como Arroz, Soja, Feijão, Cana-de-Açúcar, Trigo, Milho e entre muitos outros. A manutenção incorreta ou o tratamento inadequado devido a falta de conhecimento das doenças que atingem o solo e as plantas acarretam uma perda imensa nos lucros do setor do agronegócio, sem considerar a enorme quantia de alimentos que são desprezados e descartados por inaptidão. 

## 2.	Objetivos
O Objetivo do projeto foi desenvolver uma solução aos responsáveis das áreas de agronegócios que facilite a compreensão das patologias agrícolas encontradas utilizando um modelo de Deep Learning capaz de, através da captura de imagens, verificar a saúde de qualquer espécie vegetal. 
 Desta Forma, o setor agrícola pode aumentar a sua produtividade, velocidade e lucro devido obterem informações mais assertivas no diagnóstico da patologia de forma automatizada.

O desenvolvimento do projeto se dará nos seguintes passos:
•	Etapa 1 – Coleta de dados
•	Etapa 2 – Preparação dos Dados
•	Etapa 3 – Modelagem e criação do modelo
•	Etapa 4 – Avaliação os classificadores


## 3. Metodologias e Conceitos
Baseado em pesquisas nas quais foram aplicados algoritmos de deep learning em projetos relacionados a doenças agrícolas, este projeto foi estruturado inicialmente para ser capaz de identificar imagens com e sem patologia. Com a evolução deste projeto foi possível ir um pouco mais além, treinando assim um classificador que indicará, mediante a entrada de uma nova imagem, se a planta apresenta a patologia Oídio, ou se ela apresenta a patologia Antracnose, ou se a planta é saudável.
Será apresentado neste projeto os resultados do modelo inicial, que faz apenas a classificação de plantas com ou sem doenças, este modelo será chamado de “Modelo 1”. E também serão apresentados os resultados do modelo evoluído, que será chamado de “Modelo 2”, que além de identificar se a planta está doente ou não, ainda informa qual doença a planta tem.
Basicamente, o processo de desenvolvimento dos modelos se resume na coleta de uma imagem computacional, onde é demonstrada ao classificador que, diante de sua análise, mostra o resultado da classificação.

![image](https://github.com/EduardoMarqs/Modelo_Patologias_Agricolas/assets/26355017/cf63f6d9-e2d4-4a1e-a4ee-36311e4347ee)

## 4. Etapas
### 4.1 Entendimento do negócio
O Brasil é um dos países que mais exporta produtos agrícolas no mundo. Um país do tamanho do Brasil, com uma produtividade tão grande, e provavelmente com centenas de milhares de hectares de plantações em diversas regiões do país, é comum ter problemas relacionados a saúde de suas plantas.
Este problema é algo que os agricultores enfrentam frequentemente, pois é muito difícil controlar se existe alguma doença em suas plantações, se existir, qual ou quais doenças ela possui, e como deter essas doenças de forma rápida para que a doença não se alastre fazendo com que o prejuízo seja cada vez maior.
Dessa forma, este trabalho visa para auxiliar esses agricultores, para que eles possam ter um rápido diagnóstico de doenças agrícolas, para que seja tomada uma ação, visando a recuperação da saúde dessas plantas, além de mitigar a contaminação, e evitar maiores prejuízos.

### 4.2 Entendimento dos dados
De forma a melhorar o resultado do modelo, se faz necessário uma base de dados massiva que caracterize plantas saudáveis e plantas com características de doenças. As imagens utilizadas encontram-se disponibilizadas no site da EMBRAPA (EMBRAPA, [s.d.]). A base de dados em questão foi desenvolvida justamente para aplicações que utilizassem imagem computacional a fim de proporcionar uma solução em deep learning que facilitasse a distinção e caracterização de doenças agrícolas.
No processo de execução, ainda houve a necessidade de várias revisitações nas bases para escolher apenas as melhores imagens a serem aplicadas, pois devido a heterogeneidade presente no ambiente agrícola observou-se grande complexidade para identificar e distinguir as características da patologia. Esta etapa é essencial para o bom funcionamento do modelo. 

Para o desenvolvimento dos modelos, **foram utilizadas 5855 imagens de plantas, segregadas em três categorias: plantas saudáveis, com a doença Oídio, e com a doença Antracnose.**

### 4.3 Preparação dos dados
Durante o passo de pré-processamento das imagens, foi preciso identificar algumas características essenciais, como: posição da folha, evidência da patologia, luminosidade da imagem e densidade das plantas do fundo. 
Após as segmentações realizadas na imagem de entrada, no qual foram evidenciadas as informações relevantes para solucionar o problema, a próxima etapa dá-se na otimização dos dados, extraindo as características específicas para montar o vetor.
Das características a serem extraídas, a principal e mais relevante dentre as imagens seria a presença de manchas e lesões no limbo foliar, tanto quanto a identificação de uma folha saudável para alimentar o processo.
Uma forma de fazer a extração de características foi através de funções matemáticas como a Fourier (FFT), para diferenciar a presença de uma folha sadia de uma folha com sintomas de doença. Também existem algoritmos que podem salientar no limbo foliar cores, formatos de manchas ou lesões causadas pelas patologias na extração das características.
Foi utilizado para este projeto as propriedades da biblioteca keras, utilizando tensor flow como backend, que realiza o processamento da imagem de forma convolucionada na fase de pré-processamento. São definidos apenas os parâmetros desejados para a análise em cada camada da CNN.

Na etapa seguinte, as imagens foram segregadas para processamento da base de treinamento, validação e teste. De início foi criado um diretório para cada grupo dividos da seguinte forma: 70% dos registros para treinamento, 20% dos registros para validação, e 10% dos registros para teste, sendo que todos os grupos separados estão mesclados entre registros de plantas saudáveis e que apresentam patologia.


### 4.4 Construção do modelo
A execução do treinamento dos dois modelos foi realizada com 70% da base, sendo um total de 3850 imagens, para o tratamento das imagens foi necessário capturar todas as particularidades de cada folha, pois, no geral, todas são bem semelhantes, com o mesmo formato, mas o modelo precisa ser capaz de identificar que, uma planta verde é saudável, uma planta parecida, mas com manchas brancas, tem Oídio, e uma planta com tons marrons ou com buracos, tem Antracnose.
Segundo pesquisas realizadas, o deep learning é o método que tem conseguido melhores resultados nos projetos relacionados a processamento e identificação de imagens. (DATA SCIENCE ACADEMY, 2019b)

A partir do método de deep learning, foi necessário utilizar um algoritmo de rede neural convolucional (CNN), uma vez que este algoritmo é normalmente utilizado em projetos de processamento de imagens que necessitam de uma profunda análise e extração de características, para gerar um resultado positivo no momento da comparação. A CNN possui uma sequência de camadas que processam a imagem em pequenos lotes de pixel, conseguindo extrair o máximo de características dentro das camadas RGB da imagem. Há uma sequência de multiplicadores que varrem a imagem gerando um valor para cada pixel. A imagem é redimensionalizada e no final comparada aos resultados obtidos no treinamento, gerando no final uma predição.
A CNN foi submetida a imagens de plantas classificadas nas doenças de Oídio e Antracnose, e imagens de plantas saudáveis. No modelo 1 ela foi treinada para identificar apenas duas categorias, plantas doentes e saudáveis. Já no modelo 2, ela foi treinada para identificar se a planta está saudável, senão, identificar qual é a doença que ela possui, e no final gerar um resultado baseado nas características extraídas. 

Para alcançar o melhor resultado, a estrutura da CNN consiste em:
-	Camada de entrada: camada tem a função de fazer a leitura e padronizar a entrada da imagem na rede.
-	Camadas convolucionais: cada camada identifica e aprende as características da imagem, como padrões, bordas, cores etc.
-	Camadas pooling: nesta camada a imagem é redimensionada conforme entendimento da camada anterior.
-	Camada totalmente conectada (full connection): nesta camada é dado um peso para cada pixel de entrada.
-	Camada de saída: esta é a última camada, onde todos os pesos são verificados para retornar o valor de saída. (DATA SCIENCE ACADEMY, 2019a)
Nos dois modelos houve uma alteração na quantidade das camadas convolucional e pooling, no segundo modelo foi adicionada mais uma camada em relação ao primeiro modelo, fazendo isso atingimos um melhor resultado no processamento das imagens:


![image](https://github.com/EduardoMarqs/Modelo_Patologias_Agricolas/assets/26355017/a2252e6f-e83e-4d89-ac03-3826c1eedbf7)
Figura 2 – Etapas de processamento da CNN do Modelo 1


![image](https://github.com/EduardoMarqs/Modelo_Patologias_Agricolas/assets/26355017/3f4df258-6dd5-4c93-8b3e-695d8bf1f3c8)
Figura 3 – Etapas de processamento da CNN do Modelo 2


## 5. Resultados e Discussões
O resultado do modelo 1 teve bons resultados, baseado na acurácia nota-se que o percentual de acerto atingiu aproximadamente 100%, tanto na validação quanto no teste, já no modelo 2, o resultado teve uma acurácia um pouco menor. Isso se deve a inclusão de uma nova categoria na classificação do modelo 2, além deste modelo ter uma complexidade de classificação maior devido a uma nova categoria, também se faz necessário uma base de dados maior, para que o modelo possa ter cada vez mais êxito na segregação das categorias.

### 5.1 Validação dos modelos
Como resultado do treinamento, **o modelo 1 obteve 99,5% de acurácia e 0,03 % de loss** e, como resultado da validação, **o modelo obteve 99,8% e 0,01 % de loss.**
Já para o modelo 2, **o treinamento obteve 99,13% de acurácia e 0,03 % de loss,** e na validação, **o modelo obteve 98% de acurácia e 0,003 % de loss** na última época.
Abaixo é possível fazer um comparativo dos resultados dos dois modelos:

![image](https://github.com/EduardoMarqs/Modelo_Patologias_Agricolas/assets/26355017/816e6e7f-677c-4f9d-9f88-4e806f243415)

Também é possível fazer uma comparação com a evolução do treinamento e da validação entre os dois modelos:

![image](https://github.com/EduardoMarqs/Modelo_Patologias_Agricolas/assets/26355017/1d468f29-76cc-4717-9cf0-d0c7b73de160)
Figura 6 – Accuracy Modelo 1


![image](https://github.com/EduardoMarqs/Modelo_Patologias_Agricolas/assets/26355017/56d3c230-e1dc-432e-9282-7b5fb5a09b94)
Figura 7 - Loss Modelo 1


![image](https://github.com/EduardoMarqs/Modelo_Patologias_Agricolas/assets/26355017/e5ef590a-ad69-4b42-878e-fbcf51256b83)
Figura 8 – Accuracy Modelo 2


![image](https://github.com/EduardoMarqs/Modelo_Patologias_Agricolas/assets/26355017/d813ab13-26ef-42b7-8c03-ef44d1ae3ac5)
Figura 9 – Loss Modelo 2


### 5.2 Teste dos modelos
As imagens abaixo mostram em números os resultados dos testes realizados nos dois modelos. No teste do modelo 1, o modelo obteve 94% de precisão na classificação de plantas doentes, e 96 % de precisão na classificação de plantas saudáveis, tendo uma acurácia geral de 95%.
As tabelas de precisão mostradas mais abaixo na figura 10 e na figura 12, além de demonstrar a porcentagem da acurácia atingida, também demonstra as seguintes informações:
-	**Precision:** informa a porcentagem de acertos baseado apenas nas imagens que ele classificou para determinada categoria, por exemplo, das imagens classificadas como doentes, 94% realmente estavam doentes.
-	**Recall:** informa a porcentagem de acertos de uma categoria em relação a quantidade de vezes que a categoria apareceu na base de testes. Por exemplo, o valor de recall para plantas doentes foi de 98%, indicando que de cada 100 plantas que estão doentes, o
modelo consegue identificar 98. Pensando no negócio, este é um bom resultado, já
que é importante identificar as plantas que estão doentes.
-	F1-Score: informa a média harmônica entre precision e recall.
-	Support: informa a quantidade de imagens utilizadas.
-	Macro AVG: informa a média aritmética para precision, recall e f1-score, e a soma para o support


![image](https://github.com/EduardoMarqs/Modelo_Patologias_Agricolas/assets/26355017/79d8df10-d55d-4448-9e2c-b0b88b64136b)
Figura 10 – Precisão do Modelo 1


A matriz de confusão abaixo, mostra de forma ilustrativa a quantidade de acertos e erros no teste do primeiro modelo.

![image](https://github.com/EduardoMarqs/Modelo_Patologias_Agricolas/assets/26355017/b398174c-2c45-437b-a586-78ca9e6b81d2)
Figura 11 – Matriz de confusão Modelo 1


Já no modelo 2, conforme a tabela de precisão abaixo, fica claro que, levando em consideração a proporção de imagens utilizadas para cada uma das 3 classificações, a classificação das plantas com a patologia Antracnose foi a que mais obteve erro. Isso deve-se a pouca quantidade de imagens de plantas com essa patologia na base de imagens utilizada e, além de ter poucas imagens para treinar o algoritmo, algumas das plantas com Antracnose apresentavam pequenas manchas brancas, que fez com que o modelo fizesse uma confusão com as plantas que tem Oídio, já que a grande característica das plantas que tem a patologia Oídio é ter manchas brancas.


![image](https://github.com/EduardoMarqs/Modelo_Patologias_Agricolas/assets/26355017/bd6a21ca-ba22-4f3e-8689-a07b29359679)
Figura 12 – Precisão do Modelo 2


Pela matriz de confusão do modelo 2, é possível observar que, mesmo o modelo tendo um bom resultado de acurácia, a maior quantidade de erro foi na identificação das patologias.

![image](https://github.com/EduardoMarqs/Modelo_Patologias_Agricolas/assets/26355017/b62c2c28-59ee-4534-b698-65a2b52ed7f7)
Figura 13 – Matriz de confusão Modelo 2


## 6. Conclusão
Neste projeto foi abordado como identificar se uma planta está saudável, ou se ela contém a patologia Oídio ou Antracnose. Isso foi feito através de um modelo de deep learning, utilizando metodologia CRISP-DM, redes neurais, redes neurais convolucionais e processamento de imagem.

Os resultados mostraram que, com os modelos desenvolvidos, a classificação por imagens de plantas doentes ou saudáveis, conforme o modelo 1, pode ser feita com 95% de acurácia, e a classificação do modelo 2 pode ser realizada com 81% de acurácia, podendo-se aumentar essa eficácia com uma base de imagens maior para seja possível submeter os modelos a mais treinamentos, utilizando diversos tipos de imagens, levando em consideração todas as situações que podem influenciar uma imagem, desde o foco e qualidade da imagem, até os objetos identificados nas imagens, posição e estado de conservação das plantas.









