import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.ja.JapaneseAnalyzer;
import org.apache.lucene.analysis.ja.tokenattributes.PartOfSpeechAttribute;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.ModelSerializer;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.ConstantValueEncoder;
import org.apache.mahout.vectorizer.encoders.Dictionary;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;

import com.google.common.base.Splitter;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multiset;

/**
 */
public class LearningJapaneseText {

  private static final int FEATURES = 100000;
	public static final String PREFIX_A = "A";
	public static final String PREFIX_B = "B";

	/**
	 *
	 * Counts words
	 *
	 * @param analyzer
	 * @param words
	 * @param in
	 * @throws IOException
	 */
	private static void countWords(Analyzer analyzer, Collection<String> words,
			Reader in) throws IOException {

		TokenStream ts = analyzer.tokenStream("text", in);
		ts.reset();
		int index = 1;
		while (ts.incrementToken()) {
			String s = ts.getAttribute(CharTermAttribute.class).toString();
			String pos = ts.getAttribute(PartOfSpeechAttribute.class).getPartOfSpeech();
			if(index % 50 == 1)
			System.out.print(" " + s + "/" + pos);
			if(pos.startsWith("名詞"))
			words.add(s);
			index++;
		}
		System.out.println();
	}

	/**
	 *
	 *
	 *
	 * @param
	 * @throws IOException
	 */

	public static void main(String[] args) throws IOException {
		File base = new File("/root/chap14/jp");

		Dictionary groups = new Dictionary();
		groups.intern("A");
		groups.intern("B");

		Map<String, Set<Integer>> traceDictionary = new TreeMap<String, Set<Integer>>();
		FeatureVectorEncoder bias = new ConstantValueEncoder("Intercept");
		bias.setTraceDictionary(traceDictionary);

		OnlineLogisticRegression learningAlgorithm = new OnlineLogisticRegression(
				2, FEATURES, new L1()).alpha(1).stepOffset(1000)
				.decayExponent(0.9).lambda(3.0e-5).learningRate(20);

		List<File> files = new ArrayList<File>();
		for (File group : base.listFiles()) {
			files.addAll(Arrays.asList(group.listFiles()));
			System.out.println("group:" + group.getName());
		}

		for (File file : files) {
			BufferedReader reader = new BufferedReader(new FileReader(file));

			Multiset<String> wordsA = ConcurrentHashMultiset.create();
			Multiset<String> wordsB = ConcurrentHashMultiset.create();
			Analyzer analyzer = new JapaneseAnalyzer(Version.LUCENE_43);

			Splitter onComma = Splitter.on(",").trimResults();

			String line = reader.readLine();
			while (line != null && line.length() > 0) {
				// System.out.println("line:" + line);
				if (line.startsWith(PREFIX_A)) {
					line = Iterables.get(onComma.split(line), 1);
					StringReader in = new StringReader(line);
					countWords(analyzer, wordsA, in);
				} else if (line.startsWith(PREFIX_B)) {
					line = Iterables.get(onComma.split(line), 1);
					StringReader in = new StringReader(line);
					countWords(analyzer, wordsB, in);
				}
				line = reader.readLine();
			}

			reader.close();

			FeatureVectorEncoder encoderA = new StaticWordValueEncoder("A");
			encoderA.setProbes(2);
			encoderA.setTraceDictionary(traceDictionary);
			FeatureVectorEncoder encoderB = new StaticWordValueEncoder("B");
			encoderB.setProbes(2);
			encoderB.setTraceDictionary(traceDictionary);

			Vector v = new RandomAccessSparseVector(FEATURES);
			for (String word : wordsA.elementSet()) {
				encoderA.addToVector(word, Math.log(1 + wordsA.count(word)), v);
			}
//			for (String word : wordsB.elementSet()) {
//				encoderB.addToVector(word, Math.log(1 + wordsB.count(word)), v);
//			}
			bias.addToVector("", 1, v);

			String ng = file.getParentFile().getName();
			int actual = groups.intern(ng);
			learningAlgorithm.train(actual, v);
		}

		learningAlgorithm.close();
		ModelSerializer.writeBinary("model", learningAlgorithm);

	}

}
