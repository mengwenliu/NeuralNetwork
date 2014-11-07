import java.util.*;
import java.io.*;

public class Main {
	public static void main(String[] args) throws IOException {
		// parameters to be modified
		int numTrainExamples = 80;
		int numTestExamples = 20;
		int inputLayerSize = 350;
		int hiddenLayerSize = 80;
		int outputLayerSize = 1;
		int maxIteration = 400;
		double lambda = 1.0;
		double lr = 0.1;
		
		//train file
		Datum trainDatum = new Datum(inputLayerSize, numTrainExamples, outputLayerSize, numTrainExamples);
		trainDatum.readData("data/autism-train-80.csv");
		
		//test file
		Datum testDatum = new Datum(inputLayerSize, numTestExamples, outputLayerSize, numTestExamples);
		testDatum.readData("data/autism-test-20.csv");
        
		//testInput(datum);
		
		NeuralNetworkModel nnm = new NeuralNetworkModel();
		
		nnm.setInputLayerSize(inputLayerSize);
		nnm.setHiddenLayerSize(hiddenLayerSize);
		nnm.setOutputLayerSize(outputLayerSize);
		
		
		nnm.setLearningRate(lr);
		nnm.setLambda(lambda);
		nnm.setMaxIteration(maxIteration);
		
		nnm.train(trainDatum.getAllX(), trainDatum.getAllY());
		nnm.test(testDatum.getAllX(), testDatum.getAllY());
		
		//nnm.debug(datum.getAllX(), datum.getAllY());
	}
	
	private static void testInput(Datum datum) {
		System.out.println("All X: " + datum.getAllX().numRows() + "*" + datum.getAllX().numCols());
		System.out.println("All Y: " + datum.getAllY().numRows() + "*" + datum.getAllY().numCols());
		System.out.println();
		
		System.out.println("X_0^0: " + datum.getAllX().get(0, 0));
		System.out.println("X_1^0: " + datum.getAllX().get(1, 0));
		System.out.println("X_2^0: " + datum.getAllX().get(2, 0));
		System.out.println("X_3^0: " + datum.getAllX().get(3, 0));
		System.out.println("X_4^0: " + datum.getAllX().get(4, 0));
		System.out.println("X_5^0: " + datum.getAllX().get(5, 0));
		System.out.println("X_6^0: " + datum.getAllX().get(6, 0));
		System.out.println("X_7^0: " + datum.getAllX().get(7, 0));
		System.out.println("X_8^0: " + datum.getAllX().get(8, 0));
		System.out.println("X_9^0: " + datum.getAllX().get(9, 0));
		System.out.println("Y_0^0: " + datum.getAllY().get(0, 0));
		
		System.out.println();
		System.out.println("X_0^4: " + datum.getAllX().get(0, 4));
		System.out.println("X_1^4: " + datum.getAllX().get(1, 4));
		System.out.println("X_2^4: " + datum.getAllX().get(2, 4));
		System.out.println("X_3^4: " + datum.getAllX().get(3, 4));
		System.out.println("X_4^4: " + datum.getAllX().get(4, 4));
		System.out.println("X_5^4: " + datum.getAllX().get(5, 4));
		System.out.println("X_6^4: " + datum.getAllX().get(6, 4));
		System.out.println("X_7^4: " + datum.getAllX().get(7, 4));
		System.out.println("X_8^4: " + datum.getAllX().get(8, 4));
		System.out.println("X_9^4: " + datum.getAllX().get(9, 4));
		System.out.println("Y_0^4: " + datum.getAllY().get(0, 4));
		
		System.out.println();
		System.out.println("X_0^5: " + datum.getAllX().get(0, 5));
		System.out.println("X_1^5: " + datum.getAllX().get(1, 5));
		System.out.println("X_2^5: " + datum.getAllX().get(2, 5));
		System.out.println("X_3^5: " + datum.getAllX().get(3, 5));
		System.out.println("X_4^5: " + datum.getAllX().get(4, 5));
		System.out.println("X_5^5: " + datum.getAllX().get(5, 5));
		System.out.println("X_6^5: " + datum.getAllX().get(6, 5));
		System.out.println("X_7^5: " + datum.getAllX().get(7, 5));
		System.out.println("X_8^5: " + datum.getAllX().get(8, 5));
		System.out.println("X_9^5: " + datum.getAllX().get(9, 5));
		System.out.println("Y_0^5: " + datum.getAllY().get(0, 5));
		
	}
}

