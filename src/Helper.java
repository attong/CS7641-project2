
import func.nn.backprop.BackPropagationNetwork;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;

public class Helper{
	static double RMSE(BackPropagationNetwork network, Instance[] inst) {
		double SE = 0;
		ErrorMeasure s= new SumOfSquaresError();
		for (int i = 0; i<inst.length;i++) {
    		Instance instance = inst[i];
    		network.setInputValues(instance.getData());
    		network.run();
    		SE+=(s.value(new Instance(network.getOutputValues()), inst[i]));
		}
		return Math.pow(SE, 0.5);
	}
	private static double accuracy(BackPropagationNetwork network, Instance[] inst, ErrorMeasure measure) {
    	int correct=0,incorrect=0;
    	for (int i = 0; i<inst.length;i++) {
    		Instance instance = inst[i];
    		network.setInputValues(instance.getData());
    		network.run();
    		int pred=0, act=0;
    		double max=0;
    		for (int j = 0; j<4;j++) {
	    		double actual = instance.getLabel().getContinuous(j);
	    		double predicted = network.getOutputValues().get(j);
//	    		System.out.println(predicted);
	    		if (actual==1.0) {
//	    			System.out.println("actual: "+Integer.toString(j));
	    			act=j;
	    		}
	    		if (predicted>=max) {
	    			max=predicted;
//	    			System.out.println("predicted: "+Integer.toString(j));
	    			pred=j;
	    		}
	    		
    		}
//			System.out.println("predicted: "+Integer.toString(pred));
    		if(pred==act && pred==1) {
    			correct++;
    		}else {
    			incorrect++;
    		}
    	}
    	return correct/(float)(correct+incorrect);
	}
}