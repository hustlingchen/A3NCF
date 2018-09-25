package LDA;

import java.io.File;
import java.io.IOException;

import LDA.atmGibbsSampling.modelparameters;

public class tuningAspectNumberAndTopicNumber {
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub

		int[] topicNums = {5};
		String[] datasets = { "Patio_Lawn_and_Garden"};
		// Cell_Phones_and_Accessories
		// CDs_and_Vinyl
		// Beauty
		// Movies_and_TV
		// Clothing_Shoes_and_Jewelry
		// Musical_Instruments

		for (int t = 0; t < topicNums.length; t++) {
			int k = topicNums[t];
			for (int d = 0; d < datasets.length; d++) {
				String dataset = datasets[d];
				String originalDocsPath = "data/index_" + dataset + ".train.dat";
				String resultPath = "topicmodelresults/" + dataset + "/";
				// String parameterFile= ParameterConfig.LDAPARAMETERFILE;
				File create = new File(resultPath);
				if(!create.exists()){
					create.mkdirs();
				}

				modelparameters ldaparameters = new modelparameters();
				// getParametersFromFile(ldaparameters, parameterFile);
				Documents docSet = new Documents();
				docSet.readDocs(originalDocsPath);
				docSet.outPutIndex(resultPath);
				System.out.println("wordMap size " + docSet.tword2id.size());
				FileUtil.mkdir(new File(resultPath));
				userItemTopicModel model = new userItemTopicModel(ldaparameters);
				model.setTopicNum(k);
				model.setDataset(dataset);
				model.setResPath(resultPath);
				System.err.println("Dataset: " + dataset + "; TopicNum: " + model.getTopicNum());
				System.out.println("Saving path: " + model.getResPath());
				System.out.println("1 Initialize the model ...");
				model.initializeModel(docSet);
				System.out.println("2 Learning and Saving the model ...");
				model.inferenceModel(docSet);
				System.out.println("3 Output the final model ...");
				model.saveIteratedModel(ldaparameters.iteration, docSet);
				System.out.println("Done!");
			}
		}
	}

}
