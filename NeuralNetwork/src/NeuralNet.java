import java.util.Arrays;

public class NeuralNet {
	public int numInputs;
	public int numOutputs;
	
	public int numHiddenLayers;
	public int numHiddenNodes;
	
	double learn_rate = 0.1;
	
	public Layer[] layers;
	
	public NeuralNet(int numI, int numO, int numLayers, int nodesPerLayer, double assignWeight, double alpha) {
		numInputs = numI;
		numOutputs = numO;
		numHiddenLayers = numLayers;
		numHiddenNodes = nodesPerLayer;
		learn_rate = alpha;
		
		layers = new Layer[numLayers + 2];
		
		layers[0] = new Layer(numI);
		
		for(int i = 1; i < numLayers + 1; i++) {
			layers[i] = new Layer(layers[i - 1], nodesPerLayer);
			
			if(assignWeight == -1) {
				layers[i].AssignRandomWeights();
			}else {
				layers[i].AssignSpecificWeights(assignWeight);
			}
		}
		
		layers[numLayers + 1] = new Layer(layers[numLayers], numO);
		if(assignWeight == -1) {
			layers[numLayers + 1].AssignRandomWeights();
		}else {
			layers[numLayers + 1].AssignSpecificWeights(assignWeight);
		}
		
	}
	
	
	
	//Performs a complete forward pass of input data to output
	public double[] predictOutput(double[] input) {
		layers[0].updateLayerValues(input);
		//System.out.println("Layer 0: " + Arrays.toString(layers[0].values) + "\n");
		
		for(int i = 1; i < numHiddenLayers + 2; i++) {
			layers[i].forwardPass();
			
			//layers[i].CheckWeights();
			//System.out.println();
			
			//System.out.println("Layer " + (i) + ": " + Arrays.toString(layers[i].values) + "\n");
		}
		return _function.outputFinalizationFunction(layers[numHiddenLayers + 1].values);
	}
	
	
	public void showPrediction(double[] input) {
		layers[0].updateLayerValues(input);
		System.out.println("Inputs (Layer 0): " + Arrays.toString(layers[0].values) + "\n");
		
		for(int i = 1; i < numHiddenLayers + 2; i++) {
			layers[i].explanativeForwardPass();
			//layers[i].forwardPass();
			System.out.println("Layer Summary " + (i) + ": " + Arrays.toString(layers[i].values));
		}
		
		System.out.println(Arrays.toString(layers[numHiddenLayers + 1].values));
		System.out.println(Arrays.toString(_function.outputFinalizationFunction(layers[numHiddenLayers + 1].values)));
		System.out.println("---------");
		
		
	}
	
	public void train(double[] input, double[] expectedOutput) {
		double[] initialOutput = predictOutput(input);
		double[] error = _function.subtractArrays(initialOutput, expectedOutput);
		
		/*
		System.out.println("Output: " + Arrays.toString(initialOutput) + " |Expected: " + Arrays.toString(expectedOutput) + "| Error: " + Arrays.toString(error));
			
		*/
			
		layers[numHiddenLayers + 1].backPropagation(error, learn_rate);
		
		/*
		for(int i = 1; i < numHiddenLayers + 2; i++) {
			layers[i].CheckWeights();
		}	
		*/		
	}
	
	
	
	
}
