public class _function {
	
	//Replaced with ReLU
	public static double sigmoid(double val) {
		//return 1/(1 + Math.exp(-val));
		if (val > 0.0) {
			return val;
		}
		return 0.0;
	}
	
	//Replaced with d(ReLU)
	public static double dsigmoid(double val) {
		//return val * (1 - val);
		if (val > 0.0) {
			return 1.0;
		}
		//undefined case
		return 0.0;
	}
	
	//Function on output?
	public static double[] outputFinalizationFunction(double[] vals) {
		double[] replacer = new double[vals.length];
		for (int i = 0; i < vals.length; i++) {
			if(vals[i] > 0) {
				replacer[i] = 1;
			}else {
				replacer[i] = 0;
			}
		}
		return replacer;	
	}
	
	public static double[] lSig(double[] vals) {
		double[] nextDouble = new double[vals.length];
		for(int i = 0; i < vals.length; i++) {
			nextDouble[i] = sigmoid(vals[i]);
		}
		
		return nextDouble;
	}
	
	public static double[] lDsig(double[] vals) {
		double[] nextDouble = new double[vals.length];
		for(int i = 0; i < vals.length; i++) {
			nextDouble[i] = dsigmoid(vals[i]);
		}
		
		return nextDouble;
	}
	
	
	public static double[] subtractArrays(double[] array1, double[] array2) {
		double[] sub = new double[array1.length];
		
		for(int i = 0; i < array1.length; i++) {
			if(i >= array2.length) {
				sub[i] = array1[i];
			}else {
				sub[i] = array1[i] - array2[i];
			}
		}
		
		return sub;
	}
	
	public static double[] addArrays(double[] array1, double[] array2) {
		double[] sub = new double[array1.length];
		
		for(int i = 0; i < array1.length; i++) {
			if(i >= array2.length) {
				sub[i] = array1[i];
			}else {
				sub[i] = array1[i] + array2[i];
			}
		}
		
		return sub;
	}
	
	public static double[] multiplyArrays(double[] array1, double[] array2) {
		double[] mult = new double[array1.length];
		
		for(int i = 0; i < array1.length; i++) {
			if(i >= array2.length) {
				mult[i] = array1[i];
			}else {
				mult[i] = array1[i] * array2[i];
			}
		}
		
		return mult;
	}
	
	public static double[] multiplyConstantArrays(double[] array1, double val) {
		double[] mult = new double[array1.length];
		
		for(int i = 0; i < array1.length; i++) {
			mult[i] = array1[i] * val;
		}
		
		return mult;
	}
	
	
}
