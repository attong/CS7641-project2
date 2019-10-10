import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.io.IOException;
import java.lang.Double;
import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.TravelingSalesmanCrossOver;
import opt.example.TravelingSalesmanEvaluationFunction;
import opt.example.TravelingSalesmanRouteEvaluationFunction;
import opt.example.TravelingSalesmanSortEvaluationFunction;
import opt.ga.CrossoverFunction;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import shared.writer.CSVWriter;

public class travelSalesmanExperiment{
    private static final int N = 50;
    
	public static void main(String[] args) throws IOException, InterruptedException {
		Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();   
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        
        String[] headers = new String[9];
        headers[0]= "iterations";
        headers[1]= "RHC";
        headers[2]= "SA";
        headers[3]= "GA";
        headers[4]="MIMIC";
        headers[5]="RHC time";
        headers[6]="SA time";
        headers[7]="GA time";
        headers[8]="MIMIC time";
        CSVWriter writer = new CSVWriter("part1/traveling.csv",headers);
        try{
        	writer.open();
        } 
        catch(Exception e){
        	System.out.println(e);
        	return;
        }  
        for (int j = 0; j<5; j++) {
        	writer.nextRecord();
	        writer.write("Trial: ");
	        writer.write(Integer.toString(j));
	        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp); 
	        FixedIterationTrainer rhcfit = new FixedIterationTrainer(rhc, 10);
	        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .85, hcp);
	        FixedIterationTrainer safit = new FixedIterationTrainer(sa, 10);
	        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(100, 50, 20, gap);
	        FixedIterationTrainer gafit = new FixedIterationTrainer(ga, 10);
	        // for mimic we use a sort encoding
	//        TravelingSalesmanSortEvaluationFunction ef2 = new TravelingSalesmanSortEvaluationFunction(points);
	        int[] ranges = new int[N];
	        Arrays.fill(ranges, N);
	        odd = new  DiscreteUniformDistribution(ranges);
	        Distribution df = new DiscreteDependencyTree(.1, ranges); 
	        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
	        //400, 100
	        MIMIC mimic = new MIMIC(500, 50, pop);
	        FixedIterationTrainer mimfit = new FixedIterationTrainer(mimic, 10);
	        for (int i = 0; i<=500;i+=1) {
	        	writer.nextRecord();
		        writer.write(Integer.toString(i*10));
	        	System.out.printf("Iterations: %s \n",i);

	        	long startTime = System.nanoTime();
		        rhcfit.train();

	        	long rhctraintime = System.nanoTime()-startTime;
		        System.out.println(ef.value(rhc.getOptimal()));
		        writer.write(Double.toString(ef.value(rhc.getOptimal())));
		        
		        startTime= System.nanoTime();
		        safit.train();
		        long satraintime = System.nanoTime()-startTime;
		        writer.write(Double.toString(ef.value(sa.getOptimal())));

		        startTime= System.nanoTime();
		        gafit.train();
		        long gatraintime = System.nanoTime()-startTime;
		        System.out.println(ef.value(ga.getOptimal()));
		        writer.write(Double.toString(ef.value(ga.getOptimal())));

		        startTime= System.nanoTime();
		        mimfit.train();
		        long mimtraintime = System.nanoTime()-startTime;
		        System.out.println(ef.value(mimic.getOptimal()));
		        writer.write(Double.toString(ef.value(mimic.getOptimal())));

		        writer.write(Double.toString(rhctraintime));
		        writer.write(Double.toString(satraintime));
		        writer.write(Double.toString(gatraintime));
		        writer.write(Double.toString(mimtraintime));
	
	        }
        }

        TimeUnit.SECONDS.sleep(3);
        writer.close();
	}
}