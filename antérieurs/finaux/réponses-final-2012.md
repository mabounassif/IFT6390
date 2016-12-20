##### 1. Hyper-paramètres, capacité et sélection de modèle

1.	Le phénomène de sur-apprentissage a lieu quand le modèle apprend bien les données d'entraînement, mais prédit mal les données de validation et de teste.

2.	Le phénomène de sous-apprentissage a lieu quand le modèle apprend mal les données d'entraînement et prédit mal les données de validation et de teste.

3.	La capacité d'un algorithme définie la complexité du modèle que l'algorithme peut apprendre.

4.	(c)

5.	(d)

6.	Le dilemme bias-variance définie l'effet contradictoire qu'a le changement du bias sur la variance. Cette dernière augmente quand l'autre diminue. Donc l'augmentation de la capacité diminue le bias qui augmente la variance

7.	La valeur des hyper-paramètres qui conduit à l'erreur d'apprentissage la plus faible sur l'ensemble d'entraînement peut engendrer un sur-apprentissage. La capacité deviendra plus élevée. Si la subdivision est très petite dans l'algorithme d'histogramme le modèle aura une plus grande capacité resultant à un sur-apprentissage.

8.	

	*	$$$\lambda$$$: lambda ($$$\uparrow$$$)
	*	learning_rate (-)
	*	epoch_n ($$$\uparrow$$$)
	
9. 	Une manière efficace pour sélectionner les hyper-paramètres serait la crosse validation. Il serait d'avantage plus efficace, de partager les données en deux partie test et valid, après les avoir mélanger aléatoirement. L'optimization des paramètres se fait sur les données test et la comparaison des hyper-paramètres peut se faire sur la partie validation

10.	**func $$$B(D_n,H)$$$:  **
	&nbsp;&nbsp;&nbsp;&nbsp;$$$\lambda^\*, \hat{R}^\* $$$  
	&nbsp;&nbsp;&nbsp;&nbsp;**for** $$$\lambda$$$ **in** $$$H$$$:  
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$$$\theta = $$$**A**$$$(D_n,\lambda)$$$  
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$$$risk = \hat{R}(f_\theta, D_n)$$$  
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** $$$ risk < \hat{R}^\*: $$$  
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$$$\hat{R}^\* = risk $$$  
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$$$\lambda^\* = \lambda $$$  
	
	&nbsp;&nbsp;&nbsp;&nbsp;**return** $$$\lambda^\*$$$
	
#### 2. Concepts graphiques

1.	(6) paysage de coût ou d'erreur
2.	(4) région de décision de la classe 0
3.	(7) courbe d'apprentissage
4.	(2) ensemble des point bien classifie par $$$f_{\lambda,\theta}$$$
5.	(5) Frontière de décision
6.	(6) paysage de coût ou d'erreur
7.	(1) Frontière de decision

#### 3. Questions variées

1.	L'intérêt est de pouvoir trouver des frontières de décision complexes dans des dimensions plus grande sans besoin de calculer les transformations sur des vecteurs observations d'entrée. En gros, on peut calculer le produit scalaire de la projection sans calculer explicitement cette dernière.
2.	Regression logistique, Perceptron, Regression linéaire, classifier de plus proche moyenne, classifier de Bayes avec densités Gaulliennes à matrices de covariance identiques et SVM
3.	SVM à noyau, régression logistique à noyau, Perceptron à noyau
4.	L'algorithme boosting (adboost)
5.	+ Interprétabilité
	+ Invariabilité par translation, changement d'échelle, par transformation monotone des coordonnées
6.	On dit qu'un noeud d'un arbre de décision est "pur" quand le noeud contient que d'elements appartenant à la meme classe.
7.	Avec le boosting et bagging
8.	Clustering est semblable à une tâche de classification car les 2 taches essaye de séparer les données en group, donc de les associer à des classes. Il sont différents car la classification correspond à un apprentissage supervisé alors que le clustering est non-supervisé.
9.	La moyenne de chaque classe. $$$\mu_k \in \rm ℝ $$$ 
10.	On associe la classe dont la moyenne est la plus proche. La mesure de distance est à choisir que ce soit euclidienne ou autre
11.	    def k-means(k, D_n, iterations):  
		    mu = random_vector()
		    
		    while(true or iterations<limit):
    		    classes = {}
		    	for x in D_n:
		    		c = closest_class(d, x, mu)
		    		classes[c].append(x)
		    		
		    		new_mu = zero_vector_of_size(k)
		    		for i, class in enumerate(classes):
		    			new_mu[i] = average(class.elements)
		    			
		    		if new_mu is close(mu):
		    			break

12.	+ L'algorithme des k-moyennes est très relié à un modèle de densité gaussienne isotopique multivariée
	+ Le $$$\mu$$$ de l'epicentre de la gaussienne
	+ A l'indice de la distribution gaussienne avec la plus grande estimation de densité pour le test x
	
13.	PCA est utilise pour la compression de données, visualisation des données en 2D ou 3D et extraction de caractéristique
14.	Les "neurones cachés" et les "variables latentes" sont des valeurs non observés. La différence est comment ses valeurs sont apprise, avec la MLP c'est plutôt une descente de gradient alors qu'avec les variables latente c'est plutôt la maximization de la vraisemblance.

#### 4. Réseau de neurones et rétro-propagation du gradient
1. Un tel réseau permet d'extraire les composantes principales du test d'entrée avec une représentation de dimension réduite.
2. $$$\theta = \lbrace U, V, b, c \rbrace, b \in ℝ^{d_h}, U \in ℝ^{d_h \times d}, V \in ℝ^{d \times d_h}, c \in ℝ^d $$$
3. $$$\hat{R}\_\lambda\(f\_{\theta, D_n}\) = \sum_{i+1}^{n} L(f\_\theta(x^{(i)}), t^{(i)}) $$$
4. $$$ \theta = \theta - \epsilon \partial \frac{\hat{R_\lambda}}{\partial \theta} $$$


7. 	(a) 7. 13. 3. 14.
	(b) 5. 4. 1. 9. 8. 6. 12
	(c) 2. 15. 10. 11.