import java.util.Arrays;

public class Driver {
	public static void main(String[] args) {
		
		double[][] inputs = {
				{0, 0, 0},
				{0, 0, 1},
				{0, 1, 0},
				{0, 1, 1},
				
				{1, 0, 0},
				{1, 0, 1},
				{1, 1, 0},
				{1, 1, 1}
		};
		
		double[][] outputs = {
				{0}, 
				{0}, 
				{1}, 
				{1},
				
				{1},
				{1},
				{0},
				{0}
		};
		
		NeuralNet nw = new NeuralNet(3, 1, 1, 3, -1, 0.01);
				
		for(int i = 0; i < inputs.length; i++) {
			System.out.println("First pass: " + Arrays.toString(nw.predictOutput(inputs[i])));
		}
		
		
		for(int i = 0; i < 999999; i++) {
			for(int j = 0; j < inputs.length; j++) {
				nw.train(inputs[j], outputs[j]);
			}
		}
			
		System.out.println();
		for(int j = 0; j < inputs.length; j++) {
			double[] P_output = nw.predictOutput(inputs[j]);
			System.out.println("After training: " + Arrays.toString(P_output) + " Error: " + Arrays.toString(_function.subtractArrays(P_output, outputs[j])));
		}

		System.out.println("\nWeight Exploration!!");
		for(int i = 1; i < nw.numHiddenLayers + 2; i++) {
			nw.layers[i].CheckWeights();
		}
		
		
		//Show prediction steps!
		nw.showPrediction(inputs[7]);
		nw.showPrediction(inputs[6]);
		nw.showPrediction(inputs[5]);
		nw.showPrediction(inputs[4]);
		nw.showPrediction(inputs[3]);
		nw.showPrediction(inputs[2]);
		nw.showPrediction(inputs[1]);
		nw.showPrediction(inputs[0]);
	
	}
}
