#include <iostream>
#include <stdio.h>
#include <map>
#include <string.h>
#include <algorithm>
#include <cmath>

const int N = 10;
const int V = 100000;
using namespace std;

struct word{
	int cn;
	int l;
	double layer1[100];
	char s[100];
	double inner[N];
	char* near[N];
} vocab[V + 5];

struct node{
	int a[5];
	int num;
	node () {memset(a, -1, sizeof(a));}
};

bool operator<(node a, node b) {
	int cnt = 0;
	for (int i = 0; i < 5; ++i) {
		if (a.a[i] != -1) cnt += vocab[a.a[i]].cn;
		if (b.a[i] != -1) cnt -= vocab[b.a[i]].cn;
	}
	return cnt > 0;
}

node outputs[V + 1];
int output_num = 0;

unsigned long long hash(int x) {
	unsigned long long ret = 0;
	for (int i = 3; i < strlen(vocab[x].s); ++i) ret = ret * 290 + vocab[x].s[i];
	return ret;
}

map <unsigned long long, node> mymap;

double sqr(double x) {
	return x * x;
}

int size, layer1_size;
void findNearest(int mid_idx){
	for (int vidx = 0; vidx < V; ++vidx) {
		if (vidx == mid_idx) continue;
		double tmp = 0;
		if (vocab[vidx].l > 3 && vocab[vidx].s[0] == '(' && strcmp(vocab[mid_idx].s + 2, vocab[vidx].s + 2) == 0) continue;
		for (int k = 0; k < layer1_size; ++k) tmp += vocab[mid_idx].layer1[k] * vocab[vidx].layer1[k];
		for (int k = 0; k < 10; ++k) if (vocab[mid_idx].inner[k] < tmp) {
			for (int z = N - 1; z > k; --z) {
				vocab[mid_idx].inner[z] = vocab[mid_idx].inner[z - 1];
				vocab[mid_idx].near[z] = vocab[mid_idx].near[z - 1];
			}
			vocab[mid_idx].inner[k] = tmp;
			vocab[mid_idx].near[k] = vocab[vidx].s;
			break;
		}
	}
	for (int i = 0; i < N; ++i) {
		char* s = vocab[mid_idx].near[i];
		if (strlen(s) > 3 && s[0] == '(' && s[2] == ')') vocab[mid_idx].near[i] += 3;
	}
}

int main(){
	map<unsigned long long, node>::iterator itr;
	scanf("%d%d", &size, &layer1_size);
	for (int i = 0; i < V; i++) {
		scanf("%s%d", vocab[i].s, &(vocab[i].cn));
		vocab[i].l = strlen(vocab[i].s);
		double closev = 0;
		for (int j = 0; j < layer1_size; j++) {
			scanf("%lf", vocab[i].layer1 + j);
			closev += sqr(vocab[i].layer1[j]);
		}
		bool isWord = true;
		int l = vocab[i].l;
		int idx = 0;
		if (l > 3 && vocab[i].s[0] == '(' && vocab[i].s[2] == ')') idx = 3;
		for (; idx < l; idx++) {
			char ch = vocab[i].s[idx];
			if (!isupper(ch) && !islower(ch) && ch != '-') isWord = false;
		}
		if (!isWord) {
			i--;
			continue;
		}
		closev = sqrt(closev);
		for (int j = 0; j < layer1_size; j++) {
			vocab[i].layer1[j] /= closev;
		}
	}
	int amb_cnt = 0;
	for (int i = 0; i < V / 3; ++i) {
		if (vocab[i].s[0] != '(' || vocab[i].s[1] > '4' || vocab[i].s[1] < '0') continue;
		amb_cnt++;
		unsigned long long hs = hash(i);
		itr = mymap.find(hs);
		if (itr == mymap.end()) mymap[hs] = node();
		mymap[hs].a[vocab[i].s[1] - '0'] = i;
		memset(vocab[i].inner, 0, sizeof(vocab[i].inner));
	}
	int processed = 0;
	for (itr = mymap.begin(); itr != mymap.end(); itr++) {
		int que[5], r = 0;
		for (int i = 0; i < 5; ++i) if (itr -> second.a[i] != -1) {
			processed++;
			if (processed % 100 == 0) fprintf(stderr, "%c%d / %d processed", 13, processed, amb_cnt); fflush(stdout);
			int now = itr -> second.a[i];
			double max_cos = 0;
			for (int k = 0; k < r; ++k) {
				double x = 0;
				for (int z = 0; z < layer1_size; ++z) {
					x += vocab[now].layer1[z] * vocab[que[k]].layer1[z];
				}
				max_cos = max(max_cos, x);
			}
			if (max_cos < 0.6) {
				que[r++] = now;
			}
		}
		if (r > 1) {
			outputs[output_num].num = 0;
			for(int i = 0; i < r; ++i) {
				findNearest(que[i]);
				int dup = 0;
				for (int z = 0; z < outputs[output_num].num; z++) {
					int last = outputs[output_num].a[z];
					for (int j = 0; j < N; ++j) {
						for (int k = 0; k < N; ++k) {
							if (strcmp(vocab[last].near[j], vocab[que[i]].near[k]) == 0) dup ++;
						}
					}
				}
				if (dup < 2) {
					outputs[output_num].a[outputs[output_num].num++] = que[i];
				}
			}
			if (outputs[output_num].num >= 2)output_num++;
		}
	}
	sort(outputs, outputs + output_num);
	for (int i = 0; i < output_num; ++i) {
		int num = outputs[i].num;
		printf("%s\n", vocab[outputs[i].a[0]].s + 3);
		for (int j = 0; j < num; ++j) {
			int mid_idx = outputs[i].a[j];
			for (int k = 0; k < N; k++) {
				printf("%s ", vocab[mid_idx].near[k]);
			}
			printf("\n");
		}
	}
	return 0;
}
