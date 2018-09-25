package LDA;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class userItemTopicModel {
	int[][] doc;// sent index array
	int V, K, M;// vocabulary size, topic number, document number
	int userNum, itemNum; // number of users, number of items,
	int[][] z;// doc-term topic
	int[][] y; // doc-term y index
	float alpha_u, alpha_v; // doc-topic dirichlet prior parameter
	float beta; // topic-word dirichlet prior parameter
	float[] eta;

	int[][] nkt;// given topic k, count times of term t. K*V
	int[] nktSum;// Sum for each row in nkt

	int[] Ny0; // number of times sentence drawn from user
	int[] Ny1; // number of times sentence drawn from item
	double[] NySum;
	double[] bernpi;

	double[][] phi;// Parameters for topic-word distribution K*V
	double[][] thetaU;// Parameters for user-topic distribution
	double[][] oldThetaU; // user topic distribution
	double[][] thetaV; // parameters for item-topic distribution;

	int[][] nuk; // number of times topic for user u
	int[][] nvk; // number of times topic for item v
	int[] nukSum; // userNumber
	int[] nvkSum; // itemNum

	int iterations;// Times of iterations
	int saveStep;// The number of iterations between two saving
	int beginSaveIters;// Begin save model at this iteration
	String respath;
	String dataset = "";

	double alphaSum;
	double betaSum;
	double gammaSum;

	public userItemTopicModel(atmGibbsSampling.modelparameters modelparam) {
		// TODO Auto-generated constructor stub
		alpha_u = alpha_v = modelparam.alpha;
		beta = modelparam.beta;
		eta = modelparam.eta;

		iterations = modelparam.iteration;
		K = modelparam.topicNum;
		saveStep = modelparam.saveStep;
		beginSaveIters = modelparam.beginSaveIters;
		respath = PathConfig.LdaResultsPath;
	}

	public userItemTopicModel(float alpha, float beta, float gamma, float[] eta, int aspectNum, int iterations, int K,
			int saveStep, int beginSaveIters, String respath) {
		// TODO Auto-generated constructor stub
		this.alpha_v = this.alpha_u = alpha;
		this.beta = beta;
		this.eta = eta;

		this.iterations = iterations;
		this.K = K;
		this.saveStep = saveStep;
		this.beginSaveIters = beginSaveIters;
		this.respath = respath;
	}

	public void initializeModel(Documents docSet) {
		// TODO Auto-generated method stub
		M = docSet.docs.size();
		userNum = docSet.userNum;
		itemNum = docSet.itemNum;
		V = docSet.tword_size;

		nkt = new int[K][V]; // use to calculate phi
		nktSum = new int[K]; // use to calculate phi

		nuk = new int[userNum][K];
		nvk = new int[itemNum][K];
		nukSum = new int[userNum];
		nvkSum = new int[itemNum];

		bernpi = new double[userNum]; // use-dependent parameter pi

		phi = new double[K][V];
		thetaU = new double[userNum][K];
		oldThetaU = new double[userNum][K];
		thetaV = new double[itemNum][K];

		Ny0 = new int[userNum];
		Ny1 = new int[userNum];
		NySum = new double[userNum];

		// initialize documents index array
		doc = new int[M][]; // M = number of reviews; and then words in a review
		z = new int[M][]; // topic - word
		y = new int[M][]; // sentence from user's or from item's topic
							// distribution

		alphaSum = K * alpha_u;
		betaSum = V * beta;

		for (int m = 0; m < M; m++) {
			// Notice the limit of memory
			int N = docSet.docs.get(m).docWords.length; // number of sentences
														// in review m
			int userIdx = docSet.docs.get(m).userIdx;
			int itemIdx = docSet.docs.get(m).itemIdx;
			y[m] = new int[N];

			z[m] = new int[N];

			doc[m] = new int[N];

			for (int n = 0; n < N; n++) {
				if (Math.random() > 0.5) {
					y[m][n] = 0;
					Ny0[userIdx]++;
				} else {
					y[m][n] = 1;
					Ny1[userIdx]++;
				}
				doc[m][n] = docSet.docs.get(m).docWords[n];
				int initTopic = (int) (Math.random() * K);
				z[m][n] = initTopic;
				nkt[initTopic][doc[m][n]]++;
				nktSum[initTopic]++;

				if (y[m][n] == 0) {
					nukSum[userIdx]++;
					nuk[userIdx][initTopic]++;
				} else {
					nvkSum[itemIdx]++;
					nvk[itemIdx][initTopic]++;
				}
			}

		}

		for (int u = 0; u < userNum; u++) {
			NySum[u] = Ny0[u] + Ny1[u] + eta[0] + eta[1];
		}

	}

	public void inferenceModel(Documents docSet) throws IOException {
		// TODO Auto-generated method stub
		if (iterations < saveStep + beginSaveIters) {
			System.err.println("Error: the number of iterations should be larger than " + (saveStep + beginSaveIters));
			System.exit(0);
		}
		long startTime = System.currentTimeMillis();
		for (int i = 0; i < iterations; i++) {
			System.out.print("Iteration " + i + ": ");
			if ((i >= beginSaveIters) && (((i - beginSaveIters) % saveStep) == 0)) {
				// Saving the model
				System.out.println("Saving model at iteration " + i + " ... ");

				// Firstly update parameters
				updateEstimatedParameters();
				// whether convergence by aspect distribution
				double diff = getUserTopicDistributionConvergence();
				System.out.println("Iteration " + i + " diff: " + diff);
				// Secondly print model variables
				saveIteratedModel(i, docSet);
			}

			// Use Gibbs Sampling to update z[][]
			for (int m = 0; m < M; m++) {
				int N = docSet.docs.get(m).docWords.length;
				int userIdx = docSet.docs.get(m).userIdx;
				int itemIdx = docSet.docs.get(m).itemIdx;
				for (int n = 0; n < N; n++) {

					sampling(userIdx, itemIdx, m, n);

				}
			}
			long endTime = System.currentTimeMillis();
			long totalTime = endTime - startTime;
			startTime = endTime;
			NumberFormat formatter = new DecimalFormat("#0.00000");
			System.out.println("Execution time is " + formatter.format((totalTime) / 1000d) + " seconds");
		}

	}

	private double getUserTopicDistributionConvergence() {
		// TODO Auto-generated method stub
		double diff = 0;
		for (int i = 0; i < userNum; i++) {
			for (int j = 0; j < K; j++) {
				diff += Math.abs(oldThetaU[i][j] - thetaU[i][j]);
				oldThetaU[i][j] = thetaU[i][j];
			}
		}
		return diff;
	}

	private void sampling(int userIdx, int itemIdx, int m, int n) {
		// TODO Auto-generated method stub
		// remove topic label
		int oldY = y[m][n];
		int oldTopic = z[m][n];
		
		nkt[oldTopic][doc[m][n]]--;
		nktSum[oldTopic]--;
		if (oldY == 0) {
			Ny0[userIdx]--;
			nukSum[userIdx]--;
			nuk[userIdx][oldTopic]--;
		} else {
			Ny1[userIdx]--;
			nvkSum[itemIdx]--;
			nvk[itemIdx][oldTopic]--;
		}

		
		int newY = -1;
		int newTopic = -1;

		// Compute p(z_i = k, yi|z_-i, y_-i, w)
		double[] p = new double[2 * K];

		for (int k = 0; k < K; k++) {
			// common part
			double common_part = (nkt[k][doc[m][n]] + beta) / (nktSum[k] + betaSum);
			// first part is when y = 0;
			p[k] = (eta[0] + Ny0[userIdx]) * common_part * ((nuk[userIdx][k] + alpha_u) / (nukSum[userIdx] + alphaSum));

			// second part is when y = 1
			p[k + K] = (eta[1] + Ny1[userIdx]) * common_part * (nvk[itemIdx][k] + alpha_v)
					/ (nvkSum[itemIdx] + alphaSum);
		}

		// Sample a new topic label for w_{m, n} like roulette
		// Compute cumulated probability for p
		for (int k = 1; k < 2 * K; k++) {
			p[k] += p[k - 1];
		}
		// System.out.println(K);
		// System.out.println(Ny1 + "\t" + Ny0);
		double u = Math.random() * p[2 * K - 1]; // p[] is unnormalised
		// System.out.println(u);
		// System.out.println(p[2 * K - 1]);
		for (int t = 0; t < 2 * K; t++) {
			if (u < p[t]) {
				if (t < K) {
					newTopic = t;
					newY = 0;
				} else {
					newTopic = t - K;
					newY = 1;
				}
				break;
			}
		}
		if (newY == -1) {
			System.out.println(nukSum[userIdx]);
			System.out.println(Arrays.toString(p));
		}
		// System.err.println(newY);
		// Add new topic label for w_{m, n}
		z[m][n] = newTopic;
		y[m][n] = newY;
		nkt[newTopic][doc[m][n]]++;
		nktSum[newTopic]++;
		if (newY == 0) {
			nukSum[userIdx]++;
			nuk[userIdx][newTopic]++;
			Ny0[userIdx]++;

		} else {
			nvkSum[itemIdx]++;
			nvk[itemIdx][newTopic]++;
			Ny1[userIdx]++;
		}

	}

	private void updateEstimatedParameters() {
		// TODO Auto-generated method stub
		for (int k = 0; k < K; k++) {
			for (int t = 0; t < V; t++) {
				phi[k][t] = (nkt[k][t] + beta) / (nktSum[k] + betaSum);
			}
		}

		for (int userIdx = 0; userIdx < userNum; userIdx++) {

			for (int k = 0; k < K; k++) {
				thetaU[userIdx][k] = (nuk[userIdx][k] + alpha_u) / (nukSum[userIdx] + alphaSum);
			}

		}

		for (int itemIdx = 0; itemIdx < itemNum; itemIdx++) {

			for (int k = 0; k < K; k++) {
				thetaV[itemIdx][k] = (nvk[itemIdx][k] + alpha_v) / (nvkSum[itemIdx] + alphaSum);
			}

		}
		for (int userIdx = 0; userIdx < userNum; userIdx++) {
			bernpi[userIdx] = (eta[0] + Ny0[userIdx]) / NySum[userIdx];
		}

	}

	public void saveIteratedModel(int iters, Documents docSet) throws IOException {
		// TODO Auto-generated method stub
		// lda.params lda.phi lda.theta lda.tassign lda.twords
		// lda.params
		String resPath = this.getResPath();
		String modelName = "";
		ArrayList<String> lines = new ArrayList<String>();
		lines.add("alpha = " + alpha_u);
		lines.add("beta = " + beta);
		lines.add("eta= " + eta[0] + " " + eta[1]);
		lines.add("topicNum = " + K);
		lines.add("docNum = " + M);
		lines.add("termNum = " + V);
		lines.add("iterations = " + iterations);
		lines.add("saveStep = " + saveStep);
		lines.add("beginSaveIters = " + beginSaveIters);
		FileUtil.writeLines(resPath + modelName + ".params", lines);

		// lda.phi K*V
		BufferedWriter writer = new BufferedWriter(new FileWriter(resPath + modelName + K + ".phi"));
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < V; j++) {
				writer.write(phi[i][j] + "\t");
			}
			writer.write("\n");
		}
		writer.close();

		// lda.thetaU userNum*K
		writer = new BufferedWriter(new FileWriter(resPath + dataset + "." + K + ".user.theta"));
		for (int userIdx = 0; userIdx < userNum; userIdx++) {

			String line = String.valueOf(userIdx);
			for (int j = 0; j < K; j++) {
				line += "," + thetaU[userIdx][j];
			}
			writer.write(line.trim());
			writer.newLine();

		}
		writer.close();

		// lda.thetaV itemNum*K
		writer = new BufferedWriter(new FileWriter(resPath + dataset + "." + K + ".item.theta"));
		for (int itemIdx = 0; itemIdx < itemNum; itemIdx++) {

			String line = String.valueOf(itemIdx);
			for (int j = 0; j < K; j++) {
				line += "," + thetaV[itemIdx][j];
			}
			writer.write(line.trim());
			writer.newLine();

		}
		writer.close();



		writer = new BufferedWriter(new FileWriter(resPath + modelName + K + ".pi"));
		for (int userIdx = 0; userIdx < userNum; userIdx++) {
			writer.write(String.valueOf(userIdx) + "\t" + String.valueOf(bernpi[userIdx]));
			writer.newLine();
		}
		writer.close();

		writer = new BufferedWriter(new FileWriter(resPath + modelName + K + ".twords"));
		int topNum = 10; // Find the top 50 topic words in each topic

        for (int i = 0; i < K; i++) {
            List<Integer> tWordsIndexArray = new ArrayList<Integer>();
            for (int j = 0; j < V; j++) {
                tWordsIndexArray.add(new Integer(j));
            }
            Collections.sort(tWordsIndexArray, new userItemTopicModel.TwordsComparable(
                    phi[i]));
            writer.write("topic " + i + "\t:\t");
            for (int t = 0; t < topNum; t++) {
                writer.write(docSet.id2tword.get(tWordsIndexArray.get(t)) + "\t");
            }
            writer.write("\n");
        }
        writer.close();
	}

	public class TwordsComparable implements Comparator<Integer> {

		public double[] sortProb; // Store probability of each word in topic k

		public TwordsComparable(double[] sortProb) {
			this.sortProb = sortProb;
		}

		@Override
		public int compare(Integer o1, Integer o2) {
			// TODO Auto-generated method stub
			// Sort topic word index according to the probability of each word
			// in topic k
			if (sortProb[o1] > sortProb[o2])
				return -1;
			else if (sortProb[o1] < sortProb[o2])
				return 1;
			else
				return 0;
		}
	}

	public void setTopicNum(int topicNum) {
		this.K = topicNum;
	}

	public int getTopicNum() {
		return this.K;
	}

	public void setResPath(String resPath) {
		this.respath = resPath;
	}

	public String getResPath() {
		return this.respath;
	}

	public void setDataset(String dataset2) {
		// TODO Auto-generated method stub
		this.dataset = dataset2;
	}
}
