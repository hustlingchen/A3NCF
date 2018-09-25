package util;

import java.util.ArrayList;
import java.util.HashMap;

public class mapUpdate {
	public void listMapUpdate(HashMap<String, ArrayList<String>> map, String key, String value){
		if(map.containsKey(key)){
			ArrayList<String> list = map.get(key);
			if(!list.contains(value)){
				list.add(value);
				map.put(key, list);
			}
		}else{
			ArrayList<String> list = new ArrayList<String>();
			list.add(value);
			map.put(key, list);
		}
		
	}
	
	public void countMapUpdate(HashMap<String, Integer> map, String key){
		if(map.containsKey(key)){
			int value = map.get(key);
			value = value + 1;
			map.put(key, value);
		}else{
			map.put(key, 1);
		}
	}
}
