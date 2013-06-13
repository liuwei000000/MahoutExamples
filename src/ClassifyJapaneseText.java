import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.ja.JapaneseAnalyzer;
import org.apache.lucene.analysis.ja.tokenattributes.PartOfSpeechAttribute;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.classifier.sgd.ModelSerializer;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.DenseVector;
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
public class ClassifyJapaneseText {

  private static final int FEATURES = 100000;

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
		while (ts.incrementToken()) {
			String s = ts.getAttribute(CharTermAttribute.class).toString();
			// System.out.print(" " + s);
			String pos = ts.getAttribute(PartOfSpeechAttribute.class)
					.getPartOfSpeech();
			if (pos.startsWith("名詞"))
				words.add(s);
		}
	}

	/**
	 *
	 *
	 *
	 * @param
	 * @throws IOException
	 */

	public static void main(String[] args) throws IOException {

		File file = new File(args[0]);

		OnlineLogisticRegression deslearningAlgorithm = ModelSerializer
				.readBinary(new FileInputStream("model"),
						OnlineLogisticRegression.class);

		Map<String, Set<Integer>> traceDictionary = new TreeMap<String, Set<Integer>>();
		FeatureVectorEncoder bias = new ConstantValueEncoder("Intercept");
		bias.setTraceDictionary(traceDictionary);

		BufferedReader reader = new BufferedReader(new FileReader(file));

		Multiset<String> wordsA = ConcurrentHashMultiset.create();
		Multiset<String> wordsB = ConcurrentHashMultiset.create();
		Analyzer analyzer = new JapaneseAnalyzer(Version.LUCENE_43);

		Splitter onComma = Splitter.on(",").trimResults();

		String line = reader.readLine();
		while (line != null && line.length() > 0) {
			// System.out.println("line:" + line);
			if (line.startsWith(LearningJapaneseText.PREFIX_A)) {
				line = Iterables.get(onComma.split(line), 1);
				StringReader in = new StringReader(line);
				countWords(analyzer, wordsA, in);
			} else if (line.startsWith(LearningJapaneseText.PREFIX_B)) {
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
//		for (String word : wordsB.elementSet()) {
//			encoderB.addToVector(word, Math.log(1 + wordsB.count(word)), v);
//		}
		bias.addToVector("", 1, v);

		Dictionary groups = new Dictionary();
		groups.intern("A");
		groups.intern("B");

		Vector p = new DenseVector(2);
		deslearningAlgorithm.classifyFull(p, v);
		int iestimated = p.maxValueIndex();
		System.out.println(p.get(0));
		System.out.println(p.get(1));
		System.out.println(groups.values().get(iestimated));
	}

}
