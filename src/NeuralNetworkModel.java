import java.util.*;

import org.ejml.simple.*;

import java.lang.Object;

public class NeuralNetworkModel {
	
	protected SimpleMatrix allX, allY, theta1, theta2;
	

	
	// number of units in different layers
	private int inputLayerSize,hiddenLayerSize,outputLayerSize;
	
	// number of training examples
	private int m;
	
	// regularization term
	private double lambda;
	
	// learning rate
	private double lr;
	
    // number of iteration
	private int maxIter;
		
	// activations in each layer
	private SimpleMatrix a1, a2, p;
	
	public NeuralNetworkModel() {
		
	}
	
	public void setInputLayerSize(int inputLayerSize) {
		this.inputLayerSize = inputLayerSize;
	}
	public void setHiddenLayerSize(int hiddenLayerSize) {
		this.hiddenLayerSize = hiddenLayerSize;
	}
	public void setOutputLayerSize(int outputLayerSize) {
		this.outputLayerSize = outputLayerSize;
	}
	public void setLearningRate(double lr) {
		this.lr = lr;
	}
	public void setMaxIteration(int maxIter) {
		this.maxIter = maxIter;
	}
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}
	
	private void setA1(SimpleMatrix a1) {
		this.a1 = a1.copy();
	}
	private void setA2(SimpleMatrix a2) {
		this.a2 = a2.copy();
	}
	private void setP(SimpleMatrix p) {
		this.p = p.copy();
	}
	private void setY(SimpleMatrix allY) {
		this.allY = allY.copy();
	}
	private SimpleMatrix getA1() {
		return this.a1;
	}
	private SimpleMatrix getA2() {
		return this.a2;
	}
	private SimpleMatrix getP() {
		return this.p;
	}
	private SimpleMatrix getAllY() {
		return this.allY;
	}
	
	public void train(SimpleMatrix allX, SimpleMatrix allY) {
		theta1 = new SimpleMatrix(this.randomInitializeWeights(this.inputLayerSize, this.hiddenLayerSize));
		theta2 = new SimpleMatrix(this.randomInitializeWeights(this.hiddenLayerSize, this.outputLayerSize));
		
		this.allX = allX;
		this.allY = allY;
		this.m = allY.numCols();
		

		ArrayList<SimpleMatrix> params = new ArrayList<SimpleMatrix>();

		for (int i=0; i < this.maxIter; i++) {
			System.out.println("iteration: " + (i+1));
			this.prediction(this.theta1, this.theta2, this.allX);
			Double unRegCost = this.costFunction(this.allY, this.p);
			//System.out.println("unregularized cost: " + unRegCost.toString());
			//Double regCost = this.getRegCostFunction(unRegCost, lambda, m, theta1, theta2);
			//System.out.println("regularized cost: " + regCost.toString());
			
			ArrayList<SimpleMatrix> gradients = 
					new ArrayList<SimpleMatrix>(this.getGradients(this.theta1, this.theta2, this.inputLayerSize, this.hiddenLayerSize, this.m));
			
			//this.gradientChecking(this.theta1, this.theta2, gradients.get(0), gradients.get(1));
			//add regularization 
			//System.out.println("Add regularizaiton...");
			Double regCost = this.getRegCostFunction(unRegCost, lambda, m, theta1, theta2);
			System.out.println("regularized cost: " + regCost.toString()); 
			ArrayList<SimpleMatrix> regGradients = 
					new ArrayList<SimpleMatrix>(getRegGradients(gradients, theta1, theta2, m, lambda));
			
			// parameter estimation
			
			params.clear();
			params.add(theta1);
			params.add(theta2);
			
			ArrayList<SimpleMatrix> updateParams = new ArrayList<SimpleMatrix>(this.gradientDescent(regGradients, params, lr));
		
			
			theta1 = updateParams.get(0).copy();
			theta2 = updateParams.get(1).copy();

		}
		
	}
	
	public void test(SimpleMatrix testX, SimpleMatrix testY) {
		double precision;
		double recall;
		double f1;
		
		int tp = 0;
		int tn = 0;
		int fp = 0;
		int fn = 0;
				
		SimpleMatrix p = prediction(theta1, theta2, testX);
		//printMatrixDimension(p, "p");
		for (int i=0; i < testY.numCols(); i++) {
			if (p.get(0, i) >= 0.5) {
				p.set(0, i, 1.0);
				if (p.get(0, i) == testY.get(0, i)) {
					tp++;
				}
				else {
					fp++;
				}
			}
			else{
				p.set(0, i, 0.0);
				if (p.get(0, i) == testY.get(0, i)) {
					tn++;
				}
				else {
					fn++;
				}
			}
			
		}
		precision = (double) tp / (tp + fp);
		recall = (double) tp / (tp + fn);
		f1 = 2 * precision * recall / (precision + recall);
		System.out.println("precision: " + precision);
		System.out.println("recall: " + recall);
		System.out.println("f1: " + f1);
		
		System.out.println("Confusion Matrix");
		System.out.println("a\tb\t<-- classified as");
		System.out.println(tp + "\t" + fn + "\t| a=1");
		System.out.println(fp + "\t" + tn + "\t| b=0");
	}
	
	public void debug(SimpleMatrix x, SimpleMatrix y) {
		theta1 = debugInitializeWeights(inputLayerSize, hiddenLayerSize);
		//printMatrixDimension(theta1, "theta1");
		theta2 = debugInitializeWeights(hiddenLayerSize, outputLayerSize);
		allX = x;
		allY = y;
		m = y.numCols();
		
		ArrayList<SimpleMatrix> params = new ArrayList<SimpleMatrix>();
		
		for (int i=0; i < 350; i++) {
			System.out.println("Iteration: " + (i+1));
			prediction(theta1, theta2, allX);
			Double unRegCost = costFunction(allY, p);
			//System.out.println("unRegCost: " + unRegCost);
			ArrayList<SimpleMatrix> gradients = 
					new ArrayList<SimpleMatrix>(getGradients(theta1, theta2, inputLayerSize, hiddenLayerSize, m));
			//gradientChecking(theta1, theta2, gradients.get(0), gradients.get(1));
			
			//add regularization 
			//System.out.println("Add regularizaiton...");
			Double regCost = this.getRegCostFunction(unRegCost, lambda, m, theta1, theta2);
			System.out.println("regularized cost: " + regCost.toString()); 
			ArrayList<SimpleMatrix> regGradients = 
					new ArrayList<SimpleMatrix>(getRegGradients(gradients, theta1, theta2, m, lambda));
			//gradientChecking2(theta1, theta2, regGradients.get(0), regGradients.get(1));
			
			params.clear();
			params.add(theta1);
			params.add(theta2);
			
			ArrayList<SimpleMatrix> updateParams = new ArrayList<SimpleMatrix>(this.gradientDescent(regGradients, params, lr));
		
			
			theta1 = updateParams.get(0).copy();
			theta2 = updateParams.get(1).copy();
		}
	}
	
	private static void printMatrixDimension(SimpleMatrix sm, String str) {
		System.out.println(str  + ": " + sm.numRows() + "*" + sm.numCols());
		for(int i=0; i < sm.numRows(); i++) {
			for(int j=0; j < sm.numCols(); j++) {
				System.out.print(sm.get(i, j) + "\t");
			}
			System.out.println();
		}
	}
	
	/**
	 * Initializes the weights randomly. Some using identity.
	 */
	private SimpleMatrix randomInitializeWeights(int inLayerSize, int outLayerSize) {
		Random rgen = new Random();
		double epsilonInit = Math.sqrt(6) / Math.sqrt(inLayerSize + outLayerSize);
		//System.out.println("initial epsilon for randomInitialization: " + epsilonInit);
		SimpleMatrix w = SimpleMatrix.random(outLayerSize,inLayerSize+1, -epsilonInit, epsilonInit, rgen);
		return w;
	}

	private SimpleMatrix debugInitializeWeights(int inLayerSize, int outLayerSize) {
		SimpleMatrix w = new SimpleMatrix(outLayerSize, inLayerSize + 1);
		w.set(0.0);
		Integer k = 1;
		for(int i = 0; i < w.numCols(); i++) {
			for (int j = 0; j < w.numRows(); j++) {
				w.set(j, i, Math.sin(k.doubleValue()) / 10.0);
				//System.out.println(w.get(i,j));
				k++;
			}
		}
		return w;

	}
	
	
	private SimpleMatrix prediction(SimpleMatrix theta1, SimpleMatrix theta2, SimpleMatrix x) {
		int m = x.numCols();   //get the number of training examples
		SimpleMatrix bias = new SimpleMatrix(1, m);
	    bias.set(1.0);
	    
	    SimpleMatrix a1Bias = bias.combine(SimpleMatrix.END, 0, x);
	    
	    
//	    printMatrixDimension(x, "x");
	    //printMatrixDimension(theta1, "theta1");
	    //printMatrixDimension(a1Bias, "a1Bias");
	    
//	    System.out.println("a1Bias_0^0: " + a1Bias.get(0, 0));
//		System.out.println("a1Bias_0^1: " + a1Bias.get(0, 1));
//		System.out.println("a1Bias_0^2: " + a1Bias.get(0, 2));
//		System.out.println("a1Bias_0^3: " + a1Bias.get(0, 3));
//		System.out.println("a1Bias_0^4: " + a1Bias.get(0, 4));
//		System.out.println("a1Bias_0^5: " + a1Bias.get(0, 5));
//		System.out.println("a1Bias_0^6: " + a1Bias.get(0, 6));
//		System.out.println("a1Bias_0^7: " + a1Bias.get(0, 7));
//		System.out.println("a1Bias_0^8: " + a1Bias.get(0, 8));
//		System.out.println("a1Bias_0^9: " + a1Bias.get(0, 9));
//		System.out.println("a1Bias_0^10: " + a1Bias.get(0, 10));
//		System.out.println("a1Bias_0^11: " + a1Bias.get(0, 11));
//		System.out.println("a1Bias_0^12: " + a1Bias.get(0, 12));
//		System.out.println("a1Bias_0^13: " + a1Bias.get(0, 13));
//		System.out.println("a1Bias_0^14: " + a1Bias.get(0, 14));
//		System.out.println("a1Bias_0^15: " + a1Bias.get(0, 15));
//		System.out.println("a1Bias_0^16: " + a1Bias.get(0, 16));
//
//		
//		System.out.println("a1Bias_10^0: " + a1Bias.get(10, 0));
//		System.out.println("a1Bias_10^1: " + a1Bias.get(10, 1));
//		System.out.println("a1Bias_10^2: " + a1Bias.get(10, 2));
//		System.out.println("a1Bias_10^3: " + a1Bias.get(10, 3));
//		System.out.println("a1Bias_10^4: " + a1Bias.get(10, 4));
//		System.out.println("a1Bias_10^5: " + a1Bias.get(10, 5));
//		System.out.println("a1Bias_10^6: " + a1Bias.get(10, 6));
//		System.out.println("a1Bias_10^7: " + a1Bias.get(10, 7));
//		System.out.println("a1Bias_10^8: " + a1Bias.get(10, 8));
//		System.out.println("a1Bias_10^9: " + a1Bias.get(10, 9));
//		System.out.println("a1Bias_10^10: " + a1Bias.get(10, 10));
//		System.out.println("a1Bias_10^11: " + a1Bias.get(10, 11));
//		System.out.println("a1Bias_10^12: " + a1Bias.get(10, 12));
//		System.out.println("a1Bias_10^13: " + a1Bias.get(10, 13));
//		System.out.println("a1Bias_10^14: " + a1Bias.get(10, 14));
//		System.out.println("a1Bias_10^15: " + a1Bias.get(10, 15));
//		System.out.println("a1Bias_10^16: " + a1Bias.get(10, 16));
	    
	    
	    SimpleMatrix a2 = sigmoid(theta1.mult(a1Bias));
	    
//	    printMatrixDimension(a2, "a2");
	    
	    bias.reshape(1, a2.numCols());
	    bias.set(1.0);
	    SimpleMatrix a2Bias = bias.combine(SimpleMatrix.END, 0, a2);
	    
	    //printMatrixDimension(theta2, "theta2");
	    //printMatrixDimension(a2Bias, "a2Bias");
	    
	    SimpleMatrix p = sigmoid(theta2.mult(a2Bias));
	    
	    //printMatrixDimension(p, "p");
	    
	    this.setA1(a1Bias);
	    this.setA2(a2Bias);
	    this.setP(p);
	    
//		System.out.println("p_0^0: " + p.get(0, 0));
//		System.out.println("p_0^1: " + p.get(0, 1));
//		System.out.println("p_0^2: " + p.get(0, 2));
//		System.out.println("p_0^3: " + p.get(0, 3));

		
		
		
		

	    
		return p; //could return void
	}
	
	private Double costFunction(SimpleMatrix y, SimpleMatrix p) {
		double unregCost = 0.0;
		//int m = y.numCols();
		
		SimpleMatrix p1 = new SimpleMatrix(p.numRows(), p.numCols());
		SimpleMatrix p2 = new SimpleMatrix(p.numRows(), p.numCols());
		
		for(int i=0; i < p.numRows(); i++) {
			for(int j=0; j < p.numCols(); j++) {
				double l1 = Math.log(p.get(i,j));
				//double l2 = 1.0 - Math.log(p.get(i, j));   //wrong
				double l2 = Math.log(1.0 - p.get(i, j));
				p1.set(i, j, l1);
				p2.set(i, j, l2);
			}
		}
		
		SimpleMatrix y1 = new SimpleMatrix(y.numRows(), y.numCols());
		y1.set(1.0);
		
		//errors = - (Y .* log(a3)) - ((1 - Y) .* log(1 - a3));
		SimpleMatrix errors = y.elementMult(p1).scale(-1.0).minus(y.scale(-1.0).plus(y1).elementMult(p2));
		
		//SimpleMatrix errors1 = y.elementMult(p1).scale(-1.0);
		//printMatrixDimension(errors1, "errors1");
		
		//SimpleMatrix errors2 = y.scale(-1.0).plus(y1).elementMult(p2);
		//printMatrixDimension(errors2, "errors2");
		
		//printMatrixDimension(y, "y");
		//printMatrixDimension(errors, "errors");
		
//		System.out.println("errors^0: " + errors.get(0, 0));
//		System.out.println("errors^1: " + errors.get(0, 1));
//		System.out.println("errors^2: " + errors.get(0, 2));
//		System.out.println("errors^3: " + errors.get(0, 3));
//		System.out.println("errors^4: " + errors.get(0, 4));
//		System.out.println("errors^5: " + errors.get(0, 5));
//		System.out.println("errors^6: " + errors.get(0, 6));
//		System.out.println("errors^7: " + errors.get(0, 7));
//		System.out.println("errors^8: " + errors.get(0, 8));
//		System.out.println("errors^9: " + errors.get(0, 9));
//		System.out.println("errors^10: " + errors.get(0, 10));
//		System.out.println("errors^11: " + errors.get(0, 11));
//		System.out.println("errors^12: " + errors.get(0, 12));
//		System.out.println("errors^13: " + errors.get(0, 13));
//		System.out.println("errors^14: " + errors.get(0, 14));
//		System.out.println("errors^15: " + errors.get(0, 15));
//		System.out.println("errors^16: " + errors.get(0, 16));
		
		//System.out.println("errors sum: " + errors.elementSum());
		
		unregCost = (1.0 / Double.parseDouble(Integer.toString(this.m))) * errors.elementSum();
		
		
		return unregCost;
	}
	
	private Double getRegCostFunction(double unRegCost, double lambda, int m, SimpleMatrix theta1, SimpleMatrix theta2) {
		double regCostFunction = 0.0;
		SimpleMatrix regTheta1 = theta1.extractMatrix(0, theta1.numRows(), 1, theta1.numCols());
		SimpleMatrix regTheta2 = theta2.extractMatrix(0, theta2.numRows(), 1, theta2.numCols());
		
		
		Double penalty = (lambda / (2*m)) * (regTheta1.elementMult(regTheta1).elementSum() + regTheta2.elementMult(regTheta2).elementSum());
//		Double reg1 = regTheta1.elementMult(regTheta1).elementSum();
//		Double reg2 = regTheta2.elementMult(regTheta2).elementSum();
//		SimpleMatrix theta1s = regTheta1.elementMult(regTheta1);
//		SimpleMatrix theta2s = regTheta2.elementMult(regTheta2);
//		printMatrixDimension(theta1s, "theta1s");
//		printMatrixDimension(theta2s, "theta2s");
//		
//		System.out.println("reg1: " + reg1.toString());
//		System.out.println("reg2: " + reg2.toString());

		regCostFunction = unRegCost + penalty;
		
		return regCostFunction;
	}
	
	private ArrayList<SimpleMatrix> getGradients(SimpleMatrix theta1, SimpleMatrix theta2, int inputLayerSize, int hiddenLayerSize, int m) {
		ArrayList<SimpleMatrix> gradients = new ArrayList<SimpleMatrix>();
		
		//SimpleMatrix regTheta1 = theta1.extractMatrix(0, theta1.numCols(), 1, theta2.numCols());
		//SimpleMatrix regTheta2 = theta2.extractMatrix(0, theta2.numCols(), 1, theta2.numCols());
		
		SimpleMatrix delta3 = this.p.minus(this.allY);
		SimpleMatrix t = new SimpleMatrix(this.getA2().numRows(), this.getA2().numCols());
		t.set(1.0); 
		// delta2 = (theta1)' * delta3 .* g(z) = (theta1)' * delta3 .* a1(1-a1)
//		printMatrixDimension(theta2, "theta2");
//		printMatrixDimension(delta3, "delta3");
		
		SimpleMatrix delta2 = theta2.transpose().mult(delta3).elementMult(this.getA2().elementMult(this.getA2().scale(-1.0).plus(t)));
		delta2 = delta2.extractMatrix(1, delta2.numRows(), 0, delta2.numCols());
		
//		printMatrixDimension(delta2, "delta2");
		
		// Delta1 = delta2 * (a1)'
		SimpleMatrix allDelta1 = delta2.mult(this.a1.transpose());
		SimpleMatrix allDelta2 = delta3.mult(this.a2.transpose());
		
		SimpleMatrix gradTheta1 = allDelta1.scale(1.0 / Double.parseDouble(Integer.toString(m)));
		SimpleMatrix gradTheta2 = allDelta2.scale(1.0 / Double.parseDouble(Integer.toString(m)));
		
//		printMatrixDimension(gradTheta1, "gradTheta1");
//		printMatrixDimension(gradTheta2, "gradTheta2");
		
		gradients.add(gradTheta1);
		gradients.add(gradTheta2);
		
		
		return gradients;
	}
	
	private ArrayList<SimpleMatrix> getRegGradients(ArrayList<SimpleMatrix> unRegGradients, SimpleMatrix theta1, SimpleMatrix theta2, int m, double lambda) {
		ArrayList<SimpleMatrix> regGradients = new ArrayList<SimpleMatrix>();
		double penalty = 0.0;

		for (int i=0; i < unRegGradients.size(); i++) {
			SimpleMatrix params = new SimpleMatrix(unRegGradients.get(i));
			for (int j=0; j < unRegGradients.get(i).numRows(); j++) {
				for (int k=1; k < unRegGradients.get(i).numCols(); k++) {
					if(i == 0 )
						penalty = (lambda / m) * theta1.get(j, k);
					else
						penalty = (lambda / m) * theta2.get(j, k);
					params.set(j, k, unRegGradients.get(i).get(j, k) + penalty);
					
				}
			}
			regGradients.add(params);
		}
		
		return regGradients;
		
	}
	
	
	private void gradientChecking(SimpleMatrix theta1, SimpleMatrix theta2, SimpleMatrix gradTheta1, SimpleMatrix gradTheta2) {
		double epsilon = 0.0001;
//		printMatrixDimension(theta1, "theta1");
//		printMatrixDimension(theta2, "theta2");
		
		for (int i=0; i < theta1.numRows(); i++) {
			for (int j=0; j < theta1.numCols(); j++) {
				SimpleMatrix theta1plusE = theta1.copy();
				theta1plusE.set(i, j, theta1plusE.get(i,j) + epsilon);
				double cost1 = computeNumericalGradient(theta1plusE, theta2, this.allX, this.allY);
				SimpleMatrix theta1MinusE = theta1.copy();
				theta1MinusE.set(i, j, theta1MinusE.get(i,j) - epsilon);
				double cost2 = computeNumericalGradient(theta1MinusE, theta2, this.allX, this.allY);
				double numericalGradient = (cost1 - cost2) / (2.0 * epsilon);
				System.out.println("Theta1 " + i + ", " + j + ": " + gradTheta1.get(i, j) + "\t" + numericalGradient);
				
			}
		}
		for (int i=0; i < theta2.numRows(); i++) {
			for (int j=0; j < theta2.numCols(); j++) {
				SimpleMatrix theta2plusE = theta2.copy();
				theta2plusE.set(i, j, theta2plusE.get(i,j) + epsilon);
				double cost1 = computeNumericalGradient(theta1, theta2plusE, this.allX, this.allY);
				SimpleMatrix theta2MinusE = theta2.copy();
				theta2MinusE.set(i, j, theta2MinusE.get(i,j) - epsilon);
				double cost2 = computeNumericalGradient(theta1, theta2MinusE, this.allX, this.allY);
				double numericalGradient = (cost1 - cost2) / (2.0 * epsilon);
				System.out.println("Theta2 " + i + ", " + j + ": " + gradTheta2.get(i, j) + "\t" + numericalGradient);
				
			}
		}
	}
	
	private void gradientChecking2(SimpleMatrix theta1, SimpleMatrix theta2, SimpleMatrix gradTheta1, SimpleMatrix gradTheta2) {
		double epsilon = 0.0001;
//		printMatrixDimension(theta1, "theta1");
//		printMatrixDimension(theta2, "theta2");
		
		for (int i=0; i < theta1.numRows(); i++) {
			for (int j=0; j < theta1.numCols(); j++) {
				SimpleMatrix theta1plusE = theta1.copy();
				theta1plusE.set(i, j, theta1plusE.get(i,j) + epsilon);
				double cost1 = computeNumericalGradient2(theta1plusE, theta2, this.allX, this.allY);
				SimpleMatrix theta1MinusE = theta1.copy();
				theta1MinusE.set(i, j, theta1MinusE.get(i,j) - epsilon);
				double cost2 = computeNumericalGradient2(theta1MinusE, theta2, this.allX, this.allY);
				double numericalGradient = (cost1 - cost2) / (2.0 * epsilon);
				System.out.println("Theta1 " + i + ", " + j + ": " + gradTheta1.get(i, j) + "\t" + numericalGradient);
				
			}
		}
		for (int i=0; i < theta2.numRows(); i++) {
			for (int j=0; j < theta2.numCols(); j++) {
				SimpleMatrix theta2plusE = theta2.copy();
				theta2plusE.set(i, j, theta2plusE.get(i,j) + epsilon);
				double cost1 = computeNumericalGradient2(theta1, theta2plusE, this.allX, this.allY);
				SimpleMatrix theta2MinusE = theta2.copy();
				theta2MinusE.set(i, j, theta2MinusE.get(i,j) - epsilon);
				double cost2 = computeNumericalGradient2(theta1, theta2MinusE, this.allX, this.allY);
				double numericalGradient = (cost1 - cost2) / (2.0 * epsilon);
				System.out.println("Theta2 " + i + ", " + j + ": " + gradTheta2.get(i, j) + "\t" + numericalGradient);
				
			}
		}
	}
	
	private Double computeNumericalGradient(SimpleMatrix newTheta1, SimpleMatrix newTheta2, SimpleMatrix allX, SimpleMatrix allY) {
		SimpleMatrix p = new SimpleMatrix(this.prediction(newTheta1, newTheta2, allX));
		Double unRegCost = this.costFunction(allY, p);
		//System.out.println("unregularized cost: " + unRegCost.toString());
		return unRegCost; 
	}
		
	private Double computeNumericalGradient2(SimpleMatrix newTheta1, SimpleMatrix newTheta2, SimpleMatrix allX, SimpleMatrix allY) {
		SimpleMatrix p = new SimpleMatrix(this.prediction(newTheta1, newTheta2, allX));
		Double unRegCost = this.costFunction(allY, p);
		Double regCost = this.getRegCostFunction(unRegCost, lambda, m, newTheta1, newTheta2);
		//System.out.println("unregularized cost: " + unRegCost.toString());
		return regCost; 
	}
	
	private ArrayList<SimpleMatrix> gradientDescent(ArrayList<SimpleMatrix> gradients, ArrayList<SimpleMatrix> params, double lr) {
		double update;
		ArrayList<SimpleMatrix> updateParams = new ArrayList<SimpleMatrix>();
		for (int i=0; i < gradients.size(); i++) {
			SimpleMatrix updateParam = new SimpleMatrix(gradients.get(i));
			for (int j=0; j < gradients.get(i).numRows(); j++) {
				for (int k=0; k < gradients.get(i).numCols(); k++) {
					update = params.get(i).get(j, k) - lr * gradients.get(i).get(j, k);
					//System.out.println(update);
					updateParam.set(j, k, update);				
				}
			}
			updateParams.add(updateParam);
		}
		return updateParams;
	}
	
	
	private static SimpleMatrix sigmoid(SimpleMatrix x) {
		for(int i=0; i < x.numRows(); i++) {
			for(int j=0; j < x.numCols(); j++) {
				double g = 1.0 / (1.0 + Math.exp((-x.get(i, j))));
				x.set(i, j, g);
			}
		}
		return x;
	}

}
