import java.util.Arrays;

public class Layer {
	
	//Previous Layer
	public Layer previousLayer;
	public double[][] weights;
	public double[] biases;
	
	//the nodes of this layer
	public double[] values;
	
	//Initializes everything to 0
	public Layer(int nodes) {
		values = new double[nodes];
	}
	
	//How Hidden Layers are made
	//Nodes refer to how many nodes the hidden layer should have
	public Layer(Layer previous, int nodes) {
		previousLayer = previous;
		weights = new double[nodes][previous.values.length];
		biases = new double[nodes];
		values = new double[nodes];
	}
	
	public void updateLayerValues(double[] vals) {
		for(int i = 0; i < values.length; i++) {
			values[i] = vals[i];
		}
	}
	
	//This is the updating of this layer from last layer of this layer
	public void forwardPass() {
		for(int i = 0; i < values.length; i++) {
			
			values[i] = 0;
			
			for(int j = 0; j < previousLayer.values.length; j++) {
				values[i] += previousLayer.values[j] * weights[i][j];
			}
			
			values[i] += biases[i];
			values[i] = _function.sigmoid(values[i]);
		}
	}
	
	public void explanativeForwardPass() {
		for(int i = 0; i < values.length; i++) {
			
			values[i] = 0;
			
			for(int j = 0; j < previousLayer.values.length; j++) {
				values[i] += previousLayer.values[j] * weights[i][j];
			}
			
			values[i] += biases[i];
			
			System.out.print("Before act. func: " + values[i]);
			values[i] = _function.sigmoid(values[i]);
			System.out.print(" Func: " + values[i] + "\n	");
		}
	}
	
	//Assumes this layer as the output layer, gets the delta values for the layer before it
	public double[] getDeltasPrevLayer(double[] deltasThisLayer) {		
		
		if(previousLayer == null) {
			return null;
		}
		
		double[] deltas = new double[previousLayer.values.length];
		
		//errorsThisLayer are the errors of these layer's predicted vals as a list
		//weights for a previous not
		
		for(int i = 0; i < previousLayer.values.length; i++) {
			deltas[i] = 0;
			//Go through all the nodes in this layer using their delta values given
			for(int j = 0; j < deltasThisLayer.length; j++) {
				//Recall that the correct weight from last layer's node i to this layer's node j is weights[j][i]
				deltas[i] += weights[j][i] * deltasThisLayer[j];
			}
			deltas[i] *= _function.dsigmoid(previousLayer.values[i]);
		}
		
		//this is the new deltas of the former layer
		return deltas;
	}
	
	//Changes the weights of this layer (which are the weights that connected the previous layer to this one)
	public void weightBiasChange(double[] deltasThisLayer, double l_rate) {
		for(int i = 0; i < values.length; i++) {
			for(int j = 0; j < previousLayer.values.length; j++) {
				weights[i][j] -= l_rate * previousLayer.values[j] * deltasThisLayer[i];
			}	
			biases[i] -= l_rate * deltasThisLayer[i];
		}
	}
	
	public void backPropagation(double[] theseDeltas, double alpha) {

		if(previousLayer != null) { //if not input layer
			
			//System.out.println("\nLayer before backPropagation:");
			//CheckWeights();
			//System.out.println("\nLayer after Propagation");
			double[] prevLayerDeltas = getDeltasPrevLayer(theseDeltas);
			this.weightBiasChange(theseDeltas, alpha);
			//CheckWeights();
			//System.out.println();
			
			
			
			//System.out.println(Arrays.toString(prevLayerDeltas) + " "+ alpha);
			
			previousLayer.backPropagation(prevLayerDeltas, alpha);
		}
	}
	
	public void CheckWeights() {
		
		System.out.println("Print Biases!");
		for(int i = 0; i < values.length; i++) {
			System.out.print(biases[i] + " ");
		}
		System.out.println();
		
		System.out.print("       |");
		for(int j = 0; j < previousLayer.values.length; j++) {
			System.out.print(j + "   |");
		}
		System.out.print("\n");
		
		for(int i = 0; i < values.length; i++) {
			System.out.print("Node: " + i + " ");
			for(int j = 0; j < previousLayer.values.length; j++) {
				System.out.print(weights[i][j] + " ");
			}
			System.out.print("\n");
		}
	}
	
	public void AssignRandomWeights() {
		for(int i = 0; i < values.length; i++) {
			for(int j = 0; j < previousLayer.values.length; j++) {
				weights[i][j] = Math.random();
			}
			biases[i] = Math.random();
		}
	}
	
	public void AssignSpecificWeights(double value) {
		this.AssignRandomWeights();
		for(int i = 0; i < values.length; i++) {
			/*for(int j = 0; j < previousLayer.values.length; j++) {
				weights[i][j] = value;
			}*/
			biases[i] = value;
		}
	}
	
}
