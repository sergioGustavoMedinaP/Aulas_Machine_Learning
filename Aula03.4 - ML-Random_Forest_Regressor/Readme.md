Aula03.4: ML-Random Forest Regressor x RLM x DR
O Random Forest (Floresta Aleatória) não usa apenas uma árvore, mas sim uma "floresta" delas (centenas ou milhares). 
Ele treina cada árvore em uma parte diferente dos dados e depois tira a média de todas as previsões.

Um regressor de floresta aleatória. 
Uma floresta aleatória é um meta-estimador que ajusta uma série de regressores de árvore de decisão em várias subamostras 
do conjunto de dados e usa a média para melhorar a precisão preditiva e controlar o sobreajuste. 

As árvores na floresta usam a melhor estratégia de divisão, ou seja, equivalente a passar splitter="best"para o modelo 
subjacente DecisionTreeRegressor. 
O tamanho da subamostra é controlado pelo max_samplesparâmetro size bootstrap=True(padrão), caso contrário, todo o conjunto 
de dados é usado para construir cada árvore.

Este estimador possui suporte nativo para valores ausentes (NaNs). Durante o treinamento, o algoritmo de construção da árvore
aprende, em cada ponto de divisão, se as amostras com valores ausentes devem ser atribuídas ao filho esquerdo ou direito, 
com base no ganho potencial. Na previsão, as amostras com valores ausentes são atribuídas ao filho esquerdo ou direito, respectivamente.
Se nenhum valor ausente for encontrado para uma determinada característica durante o treinamento, as amostras com valores ausentes 
são mapeadas para o filho que tiver o maior número de amostras.

Caractersticas RFR 
Não-Linearidade: Ele captura relações complexas que uma linha reta nunca veria.
Robustez: Como é a média de muitas árvores, ele raramente sofre com um único outlier.
Sem Escalonamento: Diferente do SVR, o Random Forest não precisa que você normalize os dados. Ele lida bem com escalas 
diferentes (Dólares e Unidades) nativamente.

https://scikit--learn-org.translate.goog/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?_x_tr_sl=en&_x_tr_tl=pt&_x_tr_hl=pt&_x_tr_pto=tc

Sintaxe do RFR
class sklearn.ensemble.RandomForestRegressor ( n_estimators = 100 , * , criterion = ' squared_error' , max_depth = None , min_samples_split = 2 , 
                min_samples_leaf = 1 , min_weight_fraction_leaf = 0.0 , max_features = 1.0 , max_leaf_nodes = None , min_impurity_decrease = 0.0 , 
                bootstrap = True , oob_score = False , n_jobs = None , random_state = None , verbose = 0 , warm_start = False , ccp_alpha = 0.0 ,
                 max_samples = None , monotonic_cst = None )   
