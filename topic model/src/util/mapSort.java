package util;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;

public class mapSort {
	private FileOperator fo = null;
	private BufferedWriter bw = null;
	
	public HashMap<Integer, ArrayList<String>> intMapReverse(HashMap<String, Integer> map){
		HashMap<Integer, ArrayList<String>> sortMap = new HashMap<Integer, ArrayList<String>>();
		Iterator<String> it = map.keySet().iterator();
		while(it.hasNext()){
			String key = it.next();
			int value = map.get(key);
			if(sortMap.containsKey(value)){
				ArrayList<String> list = sortMap.get(value);
				list.add(key);
				sortMap.put(value, list);
			}else{
				ArrayList<String> list = new ArrayList<String>();
				list.add(key);
				sortMap.put(value, list);
			}
		}
		

		return sortMap;
	}
	
	public HashMap<Double, ArrayList<String>> doubleMapReverse(HashMap<String, Double> map){
		HashMap<Double, ArrayList<String>> sortMap = new HashMap<Double, ArrayList<String>>();
		Iterator<String> it = map.keySet().iterator();
		while(it.hasNext()){
			String key = it.next();
			double value = map.get(key);
			if(sortMap.containsKey(value)){
				ArrayList<String> list = sortMap.get(value);
				list.add(key);
				sortMap.put(value, list);
			}else{
				ArrayList<String> list = new ArrayList<String>();
				list.add(key);
				sortMap.put(value, list);
			}
		}
		return sortMap;
	}
	
	public void intSortMapWriter(HashMap<Integer, ArrayList<String>> intMap, File out) throws IOException{
		fo = new FileOperator();
		bw = fo.write(out);
		ArrayList<Integer> valueList = new ArrayList<Integer>();
		valueList.addAll(intMap.keySet());
		Collections.sort(valueList);
		Collections.reverse(valueList);
		
		for(int i = 0; i < valueList.size(); i++){
			int value = valueList.get(i);
			ArrayList<String> list = intMap.get(value);
			for(int j = 0; j < list.size(); j++){
				String key = list.get(j);
				bw.write(key + "\t" + value);
				bw.newLine();
			}
		}
		bw.close();
	}
	
	public void doubleSortMapWriter(HashMap<Double, ArrayList<String>> intMap, File out) throws IOException{
		fo = new FileOperator();
		bw = fo.write(out);
		ArrayList<Double> valueList = new ArrayList<Double>();
		valueList.addAll(intMap.keySet());
		Collections.sort(valueList);
		Collections.reverse(valueList);
		
		for(int i = 0; i < valueList.size(); i++){
			double value = valueList.get(i);
			ArrayList<String> list = intMap.get(value);
			for(int j = 0; j < list.size(); j++){
				String key = list.get(j);
				bw.write(key + "\t" + value);
				bw.newLine();
			}
		}
		bw.close();
	}
}
