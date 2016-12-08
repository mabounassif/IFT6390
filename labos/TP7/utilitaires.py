# coding=utf-8

import numpy
import pylab

### 
# Calcule la distance Minkowski entre un vecteur x et une matrice Y
def minkowski_mat(x,Y,p=2):
    return (numpy.sum((numpy.abs(x-Y))**p,axis=1))**(1.0/p)


##
# La fonction teste prend en entrée:
#   etiquettesTest - les étiquettes de test
#   etiquettesPred - les étiquettes prédites
# et retourne une table présentant les résultats
###
def teste(etiquettesTest, etiquettesPred):

	n_classes = max(etiquettesTest)
	conf_matrix = numpy.zeros((n_classes,n_classes))

	for (test,pred) in zip(etiquettesTest, etiquettesPred):
		conf_matrix[test-1,pred-1] += 1

	return conf_matrix
	

# fonction plot
def gridplot(classifieur,train,test,n_points=50):

    train_test = numpy.vstack((train,test))
    (min_x1,max_x1) = (min(train_test[:,0]),max(train_test[:,0]))
    (min_x2,max_x2) = (min(train_test[:,1]),max(train_test[:,1]))

    xgrid = numpy.linspace(min_x1,max_x1,num=n_points)
    ygrid = numpy.linspace(min_x2,max_x2,num=n_points)

	# calcule le produit cartesien entre deux listes
    # et met les resultats dans un array
    thegrid = numpy.array(combine(xgrid,ygrid))

    les_comptes = classifieur.compute_predictions(thegrid)
    classesPred = numpy.argmax(les_comptes,axis=1)+1

    # La grille
    # Pour que la grille soit plus jolie
    #props = dict( alpha=0.3, edgecolors='none' )
    pylab.scatter(thegrid[:,0],thegrid[:,1],c = classesPred, s=50)
	# Les points d'entrainment
    pylab.scatter(train[:,0], train[:,1], c = train[:,-1], marker = 'v', s=150)
    # Les points de test
    pylab.scatter(test[:,0], test[:,1], c = test[:,-1], marker = 's', s=150)

    ## Un petit hack, parce que la fonctionalite manque a pylab...
    h1 = pylab.plot([min_x1], [min_x2], marker='o', c = 'w',ms=5) 
    h2 = pylab.plot([min_x1], [min_x2], marker='v', c = 'w',ms=5) 
    h3 = pylab.plot([min_x1], [min_x2], marker='s', c = 'w',ms=5) 
    handles = [h1,h2,h3]
    ## fin du hack

    labels = ['grille','train','test']
    pylab.legend(handles,labels)

    pylab.axis('equal')
    pylab.show()

## http://code.activestate.com/recipes/302478/
def combine(*seqin):
    '''returns a list of all combinations of argument sequences.
for example: combine((1,2),(3,4)) returns
[[1, 3], [1, 4], [2, 3], [2, 4]]'''
    def rloop(seqin,listout,comb):
        '''recursive looping function'''
        if seqin:                       # any more sequences to process?
            for item in seqin[0]:
                newcomb=comb+[item]     # add next item to current comb
                # call rloop w/ rem seqs, newcomb
                rloop(seqin[1:],listout,newcomb)
        else:                           # processing last sequence
            listout.append(comb)        # comb finished, add to list
    listout=[]                      # listout initialization
    rloop(seqin,listout,[])         # start recursive process
    return listout

