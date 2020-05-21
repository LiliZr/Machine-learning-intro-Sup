package perceptron;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;


public class PerceptronMultiProject {
									/* Parametres */
    //Dimension des données
	public static final int Dim = 28*28+1;
    // nombre de classes
	public static int K = 12;
    // Nombre d'epoque max
    public static int EPOCHMAX = 35;
    //Dimension des données
  	public static final double eta = 0.001;

    
    
									/*les methodes*/
	
    /**fonction qui cree vecteur base canonique		
	 * @param a un entier
	 * @return un tableau d'entiers  avec la case d'indice 'a' à 1 et les autres à 0
     */
	public static int[] OneHot(int a){
		int[] t = new int [K];
		for(int i=0; i<K;i++){
			if (i==a) t[i]=1;
			else t[i]=0;
		}
		return t;
	}

	/**
	 *fonction produit scalaire
	 * @param  w un tableau de doubles
	 * @param  x un tableau de doubles
	 * @return  un double correspondant au produit scalaire entre w et x
	 */
    public static double produitScalaire(double[] w, double[] x) { 
    	double produitS=0;
    	for (int i=0; i<w.length; i++) {
    		produitS+= w[i]*x[i];
    	}
    	return produitS;
    }
   
    /**
    * fonction calcule probabilité que x appartiennent à chaque classe
    * @param x un tableau de doubles
    * @param w un tableau de doubles a deux dim
    * @return un tableau de doubles ou chaque case correspond a la probabilité que x 
    * appartienne a la classe d'indice correspondant a la case
     */
    public static double[] InfPerceptron(double[][] w, double[] x){
    	double[] probClasse = new double [K];
    	double den = 0; double num =0;
    	for (int k=0; k<K; k++){
    		den+=(double) Math.exp(produitScalaire(w[k],x));	//calcule le denominateur ' somme des exp ' 
    	}
    	for (int i=0; i<K; i++){
    		num = (double) Math.exp(produitScalaire(w[i],x));		
    		probClasse[i] = num/den;				//calcul la probabilité pour la classe i
    	}
    	return probClasse;
    }

    
   
    /**
     *fonction qui met à jour les poids du perceptron
     * @param w un tableau  a deux dim de doubles
     * @param x un tableau de doubles 
     * @param label un entier 
     * @param eta un double
     * La fonction met à jour les poids du perceptron  avec la donnée 'x' et son etiquette 'label'
     */
    public static void miseAJour(double[][] w, double[]x, int label, double eta) { 
    	double[] y = InfPerceptron(w,x);
    	int[] p = OneHot(label);		// cree le vecteur oneHot de la donnée x de classe 'label'
    	for (int k=0; k<w.length; k++){				//on met à jour chaque perceptron
	    	for (int i=0; i<w[0].length; i++) {
	    		w[k][i]=w[k][i]-(x[i]*eta*(y[k]-p[k]));
	    	}
    	}
    }


    /**
     *fonction argmax renvoie l'indice de la valeur la plus grande dans un tableau
     * @param t un tableau de double
     * @return idx un entier 
     */
    public static int argmax(double[] t){
    	int idx=0; double max=t[0];
    	for(int i=0; i<t.length; i++){
    		if (t[i]>max) {
    			idx=i;
    			max=t[i];		//maximum d un tableau
    		}
    	}
    	return idx;
    }
    

    
    /**
     *fonction qui met a jour les poids d'un ensemble de perceptrons w et calcule 
       le nombre d'erreurs commises sur un ensemble de données x
     * @param w un tableau de doubles à deux dim 'ensembles de perceptrons'
     * @param x un tableau de doubles à deux dim 'ensmble de données'
     * @param label un tableau  d'entiers 'ensemble des etiquettes des données'
     * @param eta un double
     * @return nbErreur un entier 
     */

    public static int epoque(double[][] w, double[][] x, int[] label, double eta) {
    	int nbErreur=0;
    	int ClassePredPerceptron=-1;
    	double[] a;
    	
    	for (int n=0; n<x.length; n++){
    		a = InfPerceptron(w,x[n]);
    		ClassePredPerceptron = argmax(a);			//assigne une classe a la donnée x[n] pour comparer avec sa reelle classe
    		if (ClassePredPerceptron != label[n]) {			//donnee mal classee 
    			nbErreur+=1;
    		}
    		miseAJour(w, x[n],  label[n], eta);			//met a jour l'ensemble des perceptrons
    	}
    	return nbErreur;
    	
    }


    /**
     *fonction qui calcule le nombre d'erreurs commis par le perceptron sur un ensemble de données
     * @param w un tableau de doubles a deux dim
     * @param x un tableau de doubles a deux dim
     * @param label un tableau d'entiers
     * @return nbErreur un entier
     */

    public static int nbErreur(double[][] w, double[][] x, int[] label) {
    	int nbErreur=0;
    	int ClassePredPerceptron=-1;
    	double[] a;
    	
    	for (int n=0; n<x.length; n++){
    		a = InfPerceptron(w,x[n]);
    		ClassePredPerceptron = argmax(a);		//assigne une classe a la donnee x[n]
    		if (ClassePredPerceptron != label[n]) {		//donnee mal classee
    			nbErreur+=1;
    		}	
    	}
    	return nbErreur;
    	
    }
    
    
    
    /**
	 * fonction de cout
     * @param w un tableau de doubles à deux dim
     * @param x un tableau de doubles à deux dim
     * @param label un tableau d'entiers
     * @return un double correspondaant au cout total
     */


    public static double CostFunction(double[][] w, double [][] x, int[] label){
    	double e; double somme=0;
    	int[] p;
    	double[] y;
    	for (int i=0; i<x.length; i++){		
    		e = 1;
    		p = OneHot(label[i]);			//vecteur oneHot de la donnée x[i]
    		y = InfPerceptron(w,x[i]);
    		for (int l=0; l<K; l++){
    			e*=Math.pow(y[l], p[l]);		
    		}
    		somme+=Math.log(e);			//somme le cout pour chaque donnée
    	}
    	return somme/x.length;				//calcule la moyenne
    }
    
    

    
    
	public static void main(String[] args) throws IOException {
		//cree un dossier et un sous dossier pour les resultats
		File dossier = new File("resultats/Illustrations");
		dossier.mkdirs();
		
		//!!!INDIQUER PATH DANS LA CLASSE ImagePerceptronMultiProject!!!!
		
							/********initialisation des données********/
		
        System.out.println("# initialisation des  données ! (prend moins d'une minute)");
        ImagePerceptronMultiProject d = new ImagePerceptronMultiProject();//création de l'objet pour initialiser les donnees
    	
        //initialise les trainData et label ainsi que w
        d.initialisationDonnees();
    	
    	
        FileWriter fw = new FileWriter("resultats/Resultats_nbErreur_K=12.csv"); //cree un fichier avec les resultats
        
        int iteration = 0;
        
        int nbErreurApp = nbErreur(d.w, d.trainDataApp, d.trainLabelApp);	
        int nbErreurVal = nbErreur(d.w, d.trainDataVal, d.trainLabelVal);
   		
       	double cout = CostFunction(d.w, d.trainDataApp, d.trainLabelApp);
       	
       	
       						/********programme principal perceptron********/
       	
       	fw.write("Tests avec K=12 \niteration;nbErreurApp;nbErreurVal;cout\n");
       	System.out.println("# Calcul des poids et nombre d'erreurs!(prend moins d'une minute)");
       	
       	
       	
        while (nbErreurApp>0 && iteration<EPOCHMAX) {
    		// ecrit dans le fichier les resultats
/**Q1-Q2**/ fw.write(iteration+";"+ nbErreurApp +";"+nbErreurVal+";"+cout+"\n");
    		//mise a jour perceptron et calcule les erreurs sur Apprentissage	
    		nbErreurApp = epoque(d.w, d.trainDataApp, d.trainLabelApp,eta); 
    		//calcule les erreurs sur validation
    		nbErreurVal = nbErreur(d.w, d.trainDataVal, d.trainLabelVal);	
    		//calcule cout
    		cout = CostFunction(d.w, d.trainDataApp, d.trainLabelApp);
    		
    		iteration+=1;
        }
        
/**Q2**/
        //score final sur test
        fw.write("\n Score Final sur Test:;"+nbErreur(d.w, d.trainDataTest, d.trainLabelTest)+"\n");
        fw.close();	
        
        
        						/********les resultats********/
        
        
      
        FonctionsResultatsProject r = new  FonctionsResultatsProject(); // creation de l'objet pour les resultats
        
        System.out.println("# Chargement des resultats  !\n  -Chargement Matrice de Confusion!");
        
/**Q3**/     
        			/*matrice de confusion*/
      
        // cree un fichier pour la matrice
        FileWriter fw2 = new FileWriter("resultats/Matrice_De_Confusion.csv");
        fw2.write("Matrice de confusions:\n");
        //Calcule la matrice
        double[][] matrice = r.matriceConfusion(d.w, d.trainDataVal, d.trainLabelVal);
        //enregistre la matrice dans le fichier precedent
        for(int i=0; i<matrice.length;i++){
    		for(int j=0; j<matrice[0].length;j++){
    			fw2.write(matrice[i][j]+ ";");
    		}
    		fw2.write("\n");
    	}
        fw2.close();

       
/**Q4**/
					/*Allure inferée pour chaque classe*/
        
        System.out.println("  -Enregistrement de l'allure moyenne de chaque classe inferée par le perceptron");
		
		for(int i=0; i<K; i++){
					//cree un tableau avec les images bien classées de la classe i 
			double[][] trainDataBienClassees = r.regroupeImagesBienClassees(d.w, d.trainDataVal,d.trainLabelVal,(int) matrice[i][i], i); 
					//calcule l' image Moyenne
			double[] imageMoy = r.imageMoyenne(trainDataBienClassees);
					//sauvegarde
			r.sauvegardeImg(imageMoy,i,"ImgMoyenne_De_La_Classe_");		
		}

			
/**Q5**/	
		
		
					/*recupere 5 images bien classées avec le plus mauvais taux d'inference dans l'ensemble de tests*/
		
		//cree un fichier pour les taux d'inference
		FileWriter fw3 = new FileWriter("resultats/resultats_inferences_images.csv");
		fw3.write("Inference images bien classées:\n");
        
		System.out.println("  -Enregistrement des 5 images bien classees avec le taux d'inf le plus bas!");
        
				//on recupere d'abord le nombre d'images bien classées grace a la matrice de confusion
		int nbImagesBienClassees=0;
		for(int i=0; i<K; i++) nbImagesBienClassees+=matrice[i][i];
				//on recupere les indices et les taux d'inf des images  bien classées
		double[][] imagesBienClassees = r.bienClasseesALL(d.w, d.trainDataVal, d.trainLabelVal,nbImagesBienClassees);
				//on recupere les indiceset les taux d'inf des 5 avec le plus petit taux d'inf
		double[][] MinInfBienClassees = r.minInference( imagesBienClassees, 5);	
				//sauvegarde ces 5 images
		for(int i=0; i<5; i++){
			r.sauvegardeImg(d.trainDataVal[(int)MinInfBienClassees[0][i]],d.trainLabelVal[(int)MinInfBienClassees[0][i]], "Img"+i+"_BienClassee_De_La_Classe");
			fw3.write("image"+i+";taux d'inf:;"+MinInfBienClassees[1][i]+"\n");
		}
	
		
/**Q6**/
		
					/*recupere  1a donnée la plus mal classée des classes  (F-J) */
		
		//ecrit dans le meme fichier les taux d'inference
		fw3.write("Inference images Mal classées:\n");
		System.out.println("  -Enregistrement de l'image la plus mal classée de chaque classe!");
		
		for(int i=5; i<10; i++){
					//on recup le nombre d'image mal classe pour la classe =total images-images bien classées
			int nbImagesMalClassees = (int)(r.sommeTab(matrice, i)-matrice[i][i]);	
					//images mal classées de la classe i
			double[][] imagesMalClasseesC = r.malClasseeC(d.w, d.trainDataVal, d.trainLabelVal,nbImagesMalClassees  ,i);
					//l'image avec le plus petit taux d'inf
			double[][] idxMinInfMalClasse = r.minInference(imagesMalClasseesC, 1); 
					//sauvegarde limage
			r.sauvegardeImg(d.trainDataVal[(int)idxMinInfMalClasse[0][0]],d.trainLabelVal[(int)idxMinInfMalClasse[0][0]], "Img"+i+"_MalClassee_De_la_Classe_");
			fw3.write("image"+i+";taux d'inf:;"+idxMinInfMalClasse[1][0]+"\n");
		}
		fw3.close();
		
/**Q7**/
					/*test sur les 26 classes*/
        
		System.out.println("  -Test un nouveau perceptron sur 26 classes! (prend environ deux minutes)");
        FileWriter fw4 = new FileWriter("resultats/Resultats_nbErreur_K=26.csv");
		
        K=26; ImagePerceptronMultiProject.K=26;
		EPOCHMAX=20;
		d.initialisationDonnees();
		
        iteration = 0;
        nbErreurApp = nbErreur(d.w, d.trainDataApp, d.trainLabelApp);
        nbErreurVal = nbErreur(d.w, d.trainDataVal, d.trainLabelVal);
        fw4.write("Tests avec K=26 \n iteration;nbErreurApp;nbErreurVal\n");
        
        while (nbErreurApp>0 && iteration<EPOCHMAX) {
    		fw4.write(iteration+";"+ nbErreurApp +";"+nbErreurVal+"\n");
    		//mise a jour perceptron et calcule les erreurs sur Apprentissage	
    		nbErreurApp = epoque(d.w, d.trainDataApp, d.trainLabelApp, eta); 
    		//calcule les erreurs sur validation
    		nbErreurVal = nbErreur(d.w, d.trainDataVal, d.trainLabelVal);		
    		iteration++;
        }
        fw4.write("\n Score Final sur Test:;"+nbErreur(d.w, d.trainDataTest, d.trainLabelTest)+"\n");
        fw4.close();
		
        System.out.println("# Fin");
        
        
	}

}
