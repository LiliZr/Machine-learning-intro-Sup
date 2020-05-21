package perceptron;

import java.util.Random;

import mnisttools.MnistReader;

public class ImagePerceptronMultiProject {

    String path="F:/Cours/CoursL2/S3/Java/Info216/projet/"; // !!!à  indiquer; 
    String labelDB=path+"emnist-byclass-train-labels-idx1-ubyte";
    String imageDB=path+"emnist-byclass-train-images-idx3-ubyte";
    
    
    //Dimension des données
	public static final int Dim = 28*28+1;
    // nombre de classes
	public static int K = 12;
    // Na exemples pour l'ensemble d'apprentissage
    public static final int Na = 10000; 
    // Nv exemples pour l'ensemble validation
    public static final int Nv = 1000;
    // Nt exemples pour l'ensemble de test
    public static final int Nt = 1000;

	
	
    // Generateur de nombres aleatoires
    public static int seed = 1234;
    public static Random GenRdm = new Random(seed);
	
    //variables
    
    double[][] trainDataApp;
    int[] trainLabelApp;
    int indiceMaxApp;
    double[][] trainDataVal;
    int[] trainLabelVal;
    int indiceMaxVal;
    double[][] trainDataTest;
    int[] trainLabelTest;
    int indiceMaxTest;
    double[][] w;
    MnistReader db;
    
    
    /**
     *fonction binarise une image
     * @param image un tableau a deux dim d'entiers
     * @param seuil un entier
     * @return  un tableau d'entiers a deux dimension correspondant a l'image binarisée
     */

    public  int[][] BinariserImage(int[][] image, int seuil) {
        int width = image[0].length; 
        int height = image.length; 
    	int[][] binarisee = new int [height][width];
        for (int j=0; j<height; j++) {
        	for (int i=0; i<width; i++) {
        		if (image[j][i]>seuil) binarisee[j][i]=1;		//teste par rapport au seuil
        		else binarisee[j][i]=0;
        	}
        }
        return binarisee;
    }
    
    /**
     *fonction convertImage qui convertit une image a deux dimensions a une seule dimension
     * @param image un tableau d'entiers a deux dim
     * @return image_plate un tableau d'entier unidimensionnelle
     */

    public static double[] ConvertImage(int[][] image) {
            double [] image_plate = new double[(image.length*image[0].length)+1];
            image_plate[0]=1; //premiere case=1
            for (int j=0; j<image.length; j++) {
            	for (int i=0; i<image[0].length; i++) {
            		image_plate[i+1+(j*image.length)]= image[j][i];
            		
            	}
            }
            return image_plate;
    }
    
    /**
     *fonction initialiseTraindata qui extrait nb image de db à partir de l'indice idx
     * @param db Mnistreader
     * @param idx un entier
     * @param nb un entier
     * @param traindata un tableau a deux dimensions de doubles  (initialise le tableau de donnees pris en reference)
     * @return un entier correspondant au dernier indice parcouru dans db
     */

    
    public int  initialiseTrainData(MnistReader db, int idx, int nb, double[][] trainData){
    	int i = 0;
        int j = idx; // l indice ou on commence dans db
        int classeMin=10;
        int classeMax=classeMin+K-1;
        while(i<nb){	// tant qu'on a pas extrait nb images
        	 // teste si l'etiquette est comprise entre classMin et classeMax 
        		if(db.getLabel(j)>=classeMin && db.getLabel(j)<=classeMax){
        			trainData[i]=ConvertImage(BinariserImage(db.getImage(j),128));	//extrait l image
        			i++;		//augmente le nombre d'images extraites
        		}
        	
        	j++; //on avance
        }
        return j; 
    }

    /**
     *fonction initialiseRefs qui extrait les etiquettes de nb données de db à partir de l'indice idx
     * @param db Mnistreader
     * @param idx un entier
     * @param nb un entier
     * @return un tableau d'entiers correspondant aux etiquettes 
     */
    public int[] initialiseRefs(MnistReader db, int idx, int nb) {
    	int[] trainLabel = new int[nb];
    	int i=0;
        int j =idx;	// l indice ou on commence dans db
        int classeMin=10;
        int classeMax=classeMin+K-1;
        while(i<nb){	// tant qu'on a pas extrait nb images
        	 // teste si l'etiquette est comprise entre classMin et classeMax 
        		if(db.getLabel(j)>=classeMin && db.getLabel(j)<=classeMax){
        			trainLabel[i]=db.getLabel(j)-10;        // extrait l'etiquette et la remet entre 0 et K-1
        			i++;
        		}
        	
        	j++;		//on avance
        }
        return trainLabel;
    	
    }
    
    /**
     *fonction qui initialise un perceptron
     * @param sizeW un entier 
     * @param alpha un double
     * @return w un tableau de doubles de la taille sizeW,
      dont les cases sont initialisées aleatoirement
     */

    public static double[] InitialiseW(int sizeW, double alpha) {
    	double[] w = new double[sizeW];
    	for (int i=0; i<sizeW;i++) {
    		w[i]=alpha*(GenRdm.nextFloat()-(double)0.5);
    	}
        return w;    
    }
    
    
    /**
     *fonction qui initialise un ensemble de perceptrons
     * @param sizeW un entier
     * @param alpha un double
     * @return perceptrons un tableau à K lignes correspondant à K classes et sizeW colonnes 
     * correspondant à la dimension des données
     */ 
    public static double[][] InitialiseWs(int sizeW, double alpha){
    	double [][] perceptrons = new double [K][sizeW];
    	for (int i=0; i<K; i++){		//initialise 'K' perceptron
    		perceptrons[i]= InitialiseW(sizeW, alpha);
    	}
    	return perceptrons;
    }
    
   
    
    /**
     * Fonction qui initialise les ensembles de données et leurs 
       etiquettes ainsi que les poids du perceptron
     */
    public void initialisationDonnees(){
      	
    				/* Lecteur d'image */ 
         db = new MnistReader(labelDB, imageDB);
        
         			/* initialise trainData apprentissage*/
         trainDataApp = new double[Na][Dim];
         indiceMaxApp = initialiseTrainData(db, 1, Na, trainDataApp);

         			/*initialise trainRefs apprentissage*/
         trainLabelApp = initialiseRefs(db, 1, Na);

         			/*initialise trainData validation*/
         trainDataVal = new double[Nv][Dim];
         indiceMaxVal = initialiseTrainData(db, indiceMaxApp+10, Nv, trainDataVal);
       
         			/*initialise trainRefs de validation*/
         trainLabelVal = initialiseRefs(db, indiceMaxApp+10, Nv);
         
					/*initialise trainData tests*/
         trainDataTest = new double[Nt][Dim];
         indiceMaxTest = initialiseTrainData(db, indiceMaxVal+10, Nt, trainDataTest);

					/*initialise trainRefs tests*/
         trainLabelTest = initialiseRefs(db, indiceMaxVal+10, Nt);
         
        			/*initialise w*/
         w = InitialiseWs(Dim,(double)(1.0/Dim));
    }
	
}
