OBJETIVO: Uma abordagem mais profunda sobre o algoritmo Naive Bayes

Dentro do contexto do desafio, foi permitido o uso de liberdade para o tema de preferência da pessoa para o presente desafio, no caso, escolhi o assunto do teorema Bayes e seu uso no aprendizado de máquina e para estatísticas.

Como principais objetivos, quero definir com clareza com base nas fontes que alimentei o NotebookLM gerado para o presente desafio, além de poder criar essas duas linhas de raciocínio, o Navie Bayes dentro do uso de aprendizado de máquina e dentro do ramo de estatísticas

#Listagem de fontes: Para uma melhor apuração dos dados, quis diversificar o uso das fontes, para poder elaborar não apenas o teórico sobre o teorema em si, mas também exemplos já práticos do mesmo, resultando nas seguintes linhas de pesquisa:

https://www.ibm.com/br-pt/think/topics/naive-bayes

https://repositorio.unisc.br/jspui/bitstream/11624/1994/1/Rodrigo%20de%20Abreu%20Batista.pdf

https://repositorio.ufscar.br/server/api/core/bitstreams/6679eb13-ad28-4842-817c-0b20bed117cd/content

https://youtu.be/qOIZbJ0IH2k?si=1VKn7iyvUJFDL7iM

https://youtu.be/t8RIZoQwABQ?si=u9jM_Ry7jPwYNFxi

https://youtu.be/B561LMovPHo?si=abxIQwMFMPZEwx-o


#Perguntas, prompts e dificuldades:

Como primeira pergunta que realizei após a implementação das fontes dentro do notebookLM, sendo ela "Como a suposição de independência afeta a precisão do classificador?", considerei essa como pergunta inicial por ser um dos principais pilares quando se trata do Naive Bayes, a suposição de independência define características importantes sobre esse algoritmo e como pode afetar seu desempenho comparado a outros modelos / algoritmos, algo interessante, é que a responda da IA apontou os efeitos negativos, como o esperado pela pergunta realizada, mas também elaborou de forma breve o uso de classificação supervisionada, mas no geral, os principais pontos levantados foram (frases da própria IA):

A suposição de independência afeta a precisão do classificador Naïve Bayes de formas distintas, sendo a razão fundamental para o algoritmo ser rotulado como "ingênuo" (naive) [...]

Desempenho prático satisfatório: Embora essa suposição seja frequentemente violada no mundo real — onde as variáveis costumam ser correlacionadas (como "olhos de gato" e "bigodes de gato") — o algoritmo produz resultados satisfatórios na maioria dos casos [...]

Risco de classificações incorretas: Por ser uma premissa "irrealista", o descumprimento severo da independência condicional pode levar o modelo a realizar classificações incorretas [...]

Tratamento de dados faltantes: Um efeito positivo colateral dessa independência é que o algoritmo lida muito bem com dados faltantes. Como cada coluna é calculada de forma independente, valores nulos ou desconhecidos podem ser simplesmente omitidos do cálculo da probabilidade sem comprometer o restante do processo [...]

Como parte da minha abordagem, comecei a elaborar o restante das perguntas conforme percebo trechos interessantes que a IA gera, como forma de manter a conversa para que possa chegar no ponto desejado, sendo dentro do ramo de aprendizado de máquina e para cálculos estatísticos, com isso, o próximo prompt que gerei foi "Com base no que infirmou, principalmente considerando sobre o cálculo probabilístico mais simples e viável, quais são as vantagens do Naive Bayes em relação a modelos mais complexos e considerados seguros?" onde o mesmo abordou as vantagens desse algoritmo, como a velocidade, sua fácil escalabilidade, capacidade de lidar com números massivos e incompletos de variáveis.

Como próxima pergunta, já queria saber a forma como o mesmo responde ao se tratar mais do ramo estatístico, gerando o prompt "Em que sentido o Naive Bayes é um estimador probabilístico “gerativo”?", a resposta da IA, focou primeiramente na diferença entre sua classificação focada na distribuição dos inputs, diferentemente dos classificadores discriminativos, gerando também mais alguns tópicos, sendo eles:

Modelagem da Distribuição: O algoritmo estima a probabilidade de gerar uma instância específica dado que ela pertence a uma determinada classe
Por exemplo, ele calcula qual é a probabilidade de "gerar" certas características (como o comprimento de uma pétala) dentro do "balde" de uma categoria específica de flor

Uso da Probabilidade Conjunta: Para realizar essa modelagem, o Naïve Bayes utiliza o cálculo da probabilidade conjunta das variáveis de entrada e a função de verossimilhança

Foco na Classe, não na Fronteira: Enquanto modelos como a regressão logística aprendem a fronteira de decisão entre as classes, o Naïve Bayes foca em descrever a própria classe, tratando cada atributo de forma independente para simplificar a modelagem dessa distribuição.

Em resumo, ele é gerativo porque "aprende" o perfil de cada categoria, permitindo calcular a probabilidade de que um novo dado tenha sido originado (ou gerado) por aquela classe específica

Como última pergunta, após realizar os testes de apurar o conhecimento da IA, queria reutilizar o conceito da segunda pergunta, acerca do uso do algoritmo Naive Bayes, mas dessa vez, focar especificamente em como podemos garantir que os resultados gerados podem ser úteis e mensuráveis, com isso gerei o seguinte prompt "Quais métricas definem se a precisão do classificador é satisfatória?", com a IA desenvolvendo a seguinte resposta:

A definição de uma precisão "satisfatória" para um classificador depende do contexto da aplicação e da complexidade dos dados, mas as fontes destacam a acurácia e a matriz de confusão como as principais métricas para essa avaliação

Aqui estão os critérios e métricas detalhados nas fontes:
Acurácia: É a métrica que mede a proporção de acertos totais em relação ao conjunto de amostras
De forma geral, um resultado superior a 90% é considerado muito bom em determinados casos
Em experimentos com classificadores NCM, atingiu-se 98% em conjuntos de dados fáceis, enquanto 83% a 90% foram considerados resultados satisfatórios para bases de dados médias ou difíceis

Matriz de Confusão: Esta ferramenta é essencial para entender onde o modelo está errando, pois detalha quatro indicadores críticos


Verdadeiro Positivo: Quando o modelo previu corretamente a presença de uma classe

Verdadeiro Negativo: Quando o modelo previu corretamente a ausência de uma classe

Falso Positivo: Quando o modelo indica uma classe que não existe na realidade (erro de "carimbar" algo indevidamente)

Falso Negativo: Quando o modelo falha em identificar uma classe presente (erro de omissão)

Precisão, Revocação e Medida-F1: Embora a acurácia seja a mais citada para classificação de NCM, outras métricas como a precisão (foco em evitar falsos positivos), a revocação (foco em evitar falsos negativos) e a média harmônica F1-score também são reconhecidas como métricas relevantes para validar o desempenho

Probabilidade a Posteriori (Teorema de Bayes): Em casos de eventos raros (como doenças que afetam apenas 1% da população), a acurácia isolada pode ser enganosa

Nesses cenários, a precisão satisfatória é definida pela capacidade do modelo de lidar com o desequilíbrio, evitando que um número alto de falsos positivos comprometa a utilidade do teste

Em suma, um classificador é considerado robusto quando apresenta uma acurácia alta (geralmente acima de 90%) e mantém um equilíbrio controlado entre falsos positivos e falsos negativos dentro da matriz de confusão, garantindo que o custo do erro seja aceitável para o problema de negócio em questão


