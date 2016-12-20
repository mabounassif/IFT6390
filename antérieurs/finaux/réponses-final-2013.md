#### 1 Exercice de classification
+	#### 1	
$$$P(t|x) = \frac{P(x,t)}{P(x)} = \frac{P(x|t)  \times  P(t)}{P(x)} $$$  

	$$$P(t): $$$ probabilité d'obtenir un element de la classe $$$t$$$   
$$$P(x|t): $$$ estimation de la densité de la classe $$$t$$$  
$$$P(x): $$$ la probabilité de choisir un element de l'ensemble    

	$$$P(t=1) =  0.4 $$$  
$$$P(t=2) = 0.4 $$$  
$$$P(t=3) = 0.1 $$$  
$$$P(t=4) = 0.1 $$$

	$$$P(x) = P(t=1)\hat{f_1}(x) + P(t=2)\hat{f_1}(x) + P(t=3)\hat{f_1}(x) + P(t=4)\hat{f_1}(x) = 0.9 $$$
	
	$$$ P(t|x) = [ \frac{0.5 \times 0.4}{0.9}, \frac{1.0 \times 0.4}{0.9}, \frac{2.5 \times 0.1}{0.9}, \frac{1.5 \times 0.1}{0.9} ] = [0.222, 0.444, 0.277, 0.166] $$$
	
+	#### 2 
	Vu que la class 2 a la plus haute probabilité dans ce contexte donc c'est la classe la plus probable
	
+	#### 3
	Ça sera un classifier de Bayes
	
#### 2 Arbres de décision

3.	Les arbres de decision ne sont pas des classifier linéaire.
4.	Interpretabilité, ne change pas avec les translation, changement d'echelle et par transformation monotone des coordonnées, calcule efficace
5.	Les arbres de decision sont très instable
6.	Bagging pour controller l'effet de la variance et boosting pour réduire l'instabilité.

#### 3	 Sur-apprentissage, sous-apprentissage, capacité

1.	FAUX
2.	VRAI
3.	FAUX
4.	FAUX
5.	FAUX
6.	FAUX
7.	VRAI
8.	FAUX
9.	____
10.	VRAI
11.	FAUX
