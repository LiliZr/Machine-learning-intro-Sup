package perceptron;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import javax.imageio.ImageIO;

public class FonctionsResultatsProject {
	
	public static final int SIZE=28;
	
	
								/**Fonctions utiles**/
	
	/**
	 *fonction qui cherche l'indice d'un element compris dans un tableau
     * @param tab un tableau de double
     * @param el un double
     * @return idx un entier 
	 */
	public static int chercheIdx(double[] tab, double el){
		int idx=-1;
		for(int i=0; i<tab.length; i++){
			if(tab[i]==el) idx=i;
		}
		return idx;
	}
	
	/**
	 * fonction qui somme la colonne j d'un tableau à  2dim
	 * @param t un tableau de doubles a deux dim
	 * @param j un entier
	 * @return un double
	 */
	public double sommeTab(double[][] t, int j){
		double somme=0;
		for (int i=0; i<t.length;i++){
				somme+=t[i][j];	
		}	
		return somme;
	}
	
	/**
	 *fonction qui renvoie un tableau correspondant aux 
	 indices des nb plus petites valeurs dans ce tableau
     * @param tab un tableau de double
     * @param nb un entier
     * @return un tableau d'entiers
	 */

	public int[] minimums(double[] tableau, int nb){
		ArrayList <Double> tabIntermediaire = new ArrayList<Double>();		
		int[] tabMins = new int[nb];
		int j=0;
		for(int i=0; i< tableau.length; i++){		//cree une arraylist avec les valeurs du tableau
			tabIntermediaire.add(tableau[i]);	
		}
		Collections.sort(tabIntermediaire);		//trie  l'arraylist par ordre croisssant
		for (Double d : tabIntermediaire){
			tabMins[j] = chercheIdx(tableau,d);		//on recupere dans un tableau l'indice des minimums
			j++;
			if(j==nb) break;		//si on en a recupéré nb on s'arrete
		}
		return tabMins;
	}

    /**
     *fonction convertImage qui convertit une image a 1 dim dimensions a une image a 2 dimensions
     * @param image un tableau de double
     * @return image_plate un tableau d'entier 2D
     */
  
    public double[][] convertImage2D(double[] img) {
        double [][] image_2D = new double[SIZE][SIZE];
        for (int i=0; i<image_2D.length ;i++){
        	for(int j=0; j<image_2D[0].length; j++){
        		image_2D[i][j]=img[j+1+i*image_2D.length];
        	}
        }
        return image_2D;
    }
    
    /**
     *fonction qui sauvegarde une image extraite a l'indice idx  d'un ensemble donné
     * @param un tableau de double a deux dim
     * @param idx un double
     * @param cpt un entier a rajouter dans le nom de l'image
     * @throws IOException
     */
    
    public void sauvegardeImg(double[] image, int cpt, String nomImg) throws IOException{
		String path="resultats/Illustrations/"; // chemin d'acces	
		int numberOfColumns = SIZE;
		int numberOfRows = SIZE; 
		double[][] image2D = convertImage2D(image);
		BufferedImage bimage = new BufferedImage(numberOfColumns, numberOfRows, BufferedImage.TYPE_BYTE_GRAY);
		int c;
		for(int i=0; i<SIZE; i++) {
		    for(int j=0; j<SIZE; j++) {
		    	 if ((int) image2D[i][j]==0) c=0;
		    	 else c=255; 
		         int rgb = new Color(c,c,c).getRGB();
		         bimage.setRGB(j,i,rgb);
		    }
		}

		// enregistrement
		File outputfile = new File(path+nomImg+cpt+".png");
		ImageIO.write(bimage, "png", outputfile);
    }
    
    

	
								/**Fonctions pour les resultats**/

/**Q3**/   
    
    /**
     *fonction cree une matrice de confusion
     * @param w un tableau de doubles à deux dim
     * @param x un tableau de doubles à deux dim
     * @param label un tableau d'entiers
     * @return un tableau de doubles a deux dim
     */
 
    
    public double[][] matriceConfusion(double[][] w, double[][] x, int[] label){
    	double[][] matrice = new double[w.length][w.length];
    	double [] infP;
    		for(int n=0; n<x.length; n++){
    			infP = PerceptronMultiProject.InfPerceptron(w,x[n]);
    			matrice[PerceptronMultiProject.argmax(infP)][label[n]]+=1;
    		}
    	return matrice;   	
    }

    
    
    
/**Q4**/
    
    
    /**
     *fonction qui renvoie un tableau correspondant aux images de x,
       de la classe "classe"  et bien classées par w
	  * @param w un tableau de doubles à deux dim
	  * @param x un tableau de doubles à deux dim
	  * @param label un tableau d'entiers
	  * @param nb un entier
	  * @param classe un entier
	  * @return images un tableau de doubles a deux dim
     */

	public double[][] regroupeImagesBienClassees(double[][] w, double[][] x,int[] label, int nb, int classe){
		double[][] images = new double[nb][PerceptronMultiProject.Dim];
		double[] p;
		int j=0;
	 	for(int i=0; i<x.length; i++){
	 		p = PerceptronMultiProject.InfPerceptron(w,x[i]);
			if (PerceptronMultiProject.argmax(p)==label[i] &&  label[i]==classe){ 	
				images[j]=x[i];
				j++;
			}
			if (j==nb) break;
	 	}
	 	return images;
	}
	
	/**
	 * fonction renvoie l'image moyenne sur un ensemble d'images
	 * @param ensembleImages un tableau de double à deux dim
	 * @return un tableau de double correspondant a l'image moyenne
	 */
	
	public double[] imageMoyenne(double[][] ensembleImages){
		double[] t = new double[PerceptronMultiProject.Dim];
		double nb=0;
		for (int i=0; i<t.length; i++){
			nb = sommeTab(ensembleImages,i);
			if ((nb/ensembleImages.length)>=0.5) t[i]=1;
			else t[i]=0;
		}
		return t;
	}

	
	
	
/**Q5**/
    
    /**
     *fonction qui renvoie un tableau d'informations sur Toutes les données 
		     bien classées 
       la 1ere ligne correspond aux indices de ces données dans l'ensemble x
       la 2eme ligne au taux d'inference du perceptron sur ces données 
      
     * @param w un tableau de doubles à deux dim
     * @param x un tableau de doubles à deux dim
     * @param label un tableau d'entiers
     * @param nb un entier
     * @return un tableau de doubles a deux dim 
     */
    
	public double[][] bienClasseesALL(double[][]w, double[][]x, int[] label, int nb){
		double[] p;
		int j=0;
		double [][] tableau = new double [2][nb]; //cree un tableau selon le nombre d'images mal classées 	
	 	for(int i=0; i<x.length; i++){
	 		p = PerceptronMultiProject.InfPerceptron(w,x[i]);
			if (PerceptronMultiProject.argmax(p)==label[i]){ 		//si bien classée
		 		tableau[0][j] = i;					//on recupere l'indice
		 		tableau[1][j] = p[label[i]];				//on recupere le taux d'inference
		 		j++;
	 		}
			if(j==nb) break;		//si on a tout récuperé c'est bon
	 	}
	 	return tableau;
		
	}
	
	
    /**
     *fonction qui renvoie un tableau d'informations sur nb données avec le plus petit taux d'inference
      1ere ligne correspond aux indices de ces donnees
      et la deuxieme ligne au taux d'inferecences de ces images
     * @param tableau un tableau de doubles 
     * @param n un entier
     * @return tableau d'entiers
     */
 
    public double[][] minInference(double[][] imagesClassees, int nb){
    	double[][] tab = new double[2][nb];
    	int[] mins = minimums(imagesClassees[1],nb);
    	for(int i=0; i<mins.length; i++ ){
    		tab[0][i] = imagesClassees[0][mins[i]];		//on recupere l'indice
    		tab[1][i] = imagesClassees[1][mins[i]];
    	}
    	return tab;	
    }
	
    
/**Q6**/
    
	 /**
	  *fonction qui renvoie un tableau d'informations sur les données 
		 mal classées de la classe "classe"
		    
		  la 1ere ligne correspond aux indices de ces données dans l'ensemble x
		  la 2eme ligne au taux d'inference du perceptron sur ces données 
		   
	  * @param w un tableau de doubles à deux dim
	  * @param x un tableau de doubles à deux dim
	  * @param label un tableau d'entiers
	  * @param nb un entier
	  * @return un tableau de doubles a deux dim 
	  */
	 
	public double[][] malClasseeC(double[][] w, double[][] x, int[] label, int nb, int classe){
		double[] p;
		int j=0;
		double [][] tableau = new double [2][nb]; //cree un tableau selon le nombre d'images mal classées 	
	 	for(int i=0; i<x.length; i++){
	 		p = PerceptronMultiProject.InfPerceptron(w,x[i]);
			if (PerceptronMultiProject.argmax(p)!=label[i] && label[i]==classe){ 		//si mal classée
		 		tableau[0][j] = i;					//on recupere l'indice
		 		tableau[1][j] = p[label[i]];				//on recupere le taux d'inference
		 		j++;
	 		}
			if(j==nb) break;		//si on a tout réuperé c'est bon
	 	}
	 	return tableau;
			
	}
	
    
}
