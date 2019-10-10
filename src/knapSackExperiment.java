//package opt.test;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import shared.writer.CSVWriter;

/**
 * A test of the knapsack problem
 *
 * Given a set of items, each with a weight and a value, determine the number of each item to include in a
 * collection so that the total weight is less than or equal to a given limit and the total value is as
 * large as possible.
 * https://en.wikipedia.org/wiki/Knapsack_problem
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class knapSackExperiment {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 100;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum value for a single element */
    private static final double MAX_VALUE = 50;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum weight for the knapsack */
    private static final double MAX_KNAPSACK_WEIGHT =
         MAX_WEIGHT * NUM_ITEMS * COPIES_EACH * .4;

    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) throws IOException, InterruptedException{
        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] values = new double[NUM_ITEMS];
        double[] weights = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            values[i] = random.nextDouble() * MAX_VALUE;
            weights[i] = random.nextDouble() * MAX_WEIGHT;
        }
        int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);

        EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);

        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
//        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
//        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
//        fit.train();
//        System.out.println(ef.value(rhc.getOptimal()));
//        
//        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
//        fit = new FixedIterationTrainer(sa, 200000);
//        fit.train();
//        System.out.println(ef.value(sa.getOptimal()));
//        
//        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 25, gap);
//        fit = new FixedIterationTrainer(ga, 2000);
//        fit.train();
//        System.out.println(ef.value(ga.getOptimal()));
//        
//        MIMIC mimic = new MIMIC(200, 100, pop);
//        fit = new FixedIterationTrainer(mimic, 1000);
//        fit.train();
//        System.out.println(ef.value(mimic.getOptimal()));
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
        CSVWriter writer = new CSVWriter("part1/knapsack.csv",headers);
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
	        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
	        FixedIterationTrainer safit = new FixedIterationTrainer(sa, 10);
	        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(100, 50, 20, gap);
	        FixedIterationTrainer gafit = new FixedIterationTrainer(ga, 10);
	        // for mimic we use a sort encoding
//	        Arrays.fill(ranges, N);
	        odd = new  DiscreteUniformDistribution(ranges);
//	        Distribution df = new DiscreteDependencyTree(.1, ranges); 
//	        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
	        //400, 100
	        MIMIC mimic = new MIMIC(200, 100, pop);
	        FixedIterationTrainer mimfit = new FixedIterationTrainer(mimic, 10);
	        long rhctraintime = 0;
	        long satraintime = 0;
	        long gatraintime = 0;
	        long mimtraintime = 0;
	        for (int i = 0; i<=500;i+=1) {
	        	writer.nextRecord();
		        writer.write(Integer.toString(i*10));
	        	System.out.printf("Iterations: %s \n",i);

	        	long startTime = System.nanoTime();
		        rhcfit.train();

	        	rhctraintime += System.nanoTime()-startTime;
		        System.out.println(ef.value(rhc.getOptimal()));
		        writer.write(Double.toString(ef.value(rhc.getOptimal())));
		        
		        startTime= System.nanoTime();
		        safit.train();
		        satraintime += System.nanoTime()-startTime;
		        writer.write(Double.toString(ef.value(sa.getOptimal())));
		        System.out.println(ef.value(sa.getOptimal()));

		        startTime= System.nanoTime();
		        gafit.train();
		        gatraintime += System.nanoTime()-startTime;
		        writer.write(Double.toString(ef.value(ga.getOptimal())));
		        System.out.println(Double.toString(ef.value(ga.getOptimal())));

		        startTime= System.nanoTime();
		        mimfit.train();
		        mimtraintime += System.nanoTime()-startTime;
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
