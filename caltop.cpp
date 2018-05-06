#include <iostream>
#include <stdio.h>
#include <map>
#include <string.h>
#include <algorithm>

const int N = 10;
const int V = 100000;
using namespace std;

struct word{
	int cn;
	double layer1[100];
	char s[100];
	double inner[N];
	int idx[N];
} vocab[V + 5];

struct node{
	int a[5];
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

int main(){
	map<unsigned long long, node>::iterator itr;
	int size, layer1_size;
	scanf("%d%d", &size, &layer1_size);
	for (int i = 0; i < V; i++) {
		scanf("%s%d", vocab[i].s, &(vocab[i].cn));
		for (int j = 0; j < layer1_size; ++j) scanf("%lf", vocab[i].layer1 + j);
	}
	for (int i = 0; i < V; ++i) {
		if (i % 1000 == 0) fprintf(stderr, "%c %dK", 13, i / 1000); fflush(stdout);
		if (vocab[i].s[0] != '(' || vocab[i].s[1] > '4' || vocab[i].s[1] < '0') continue;
		unsigned long long hs = hash(i);
		itr = mymap.find(hs);
		if (itr == mymap.end()) mymap[hs] = node();
		mymap[hs].a[vocab[i].s[1] - '0'] = i;
		memset(vocab[i].inner, 0, sizeof(vocab[i].inner));
		memset(vocab[i].idx, -1, sizeof(vocab[i].idx));
		for (int j = 0; j < V; ++j) {
			if (j == i || (strlen(vocab[j].s) > 3 && vocab[j].s[0] == '(' && vocab[j].s[2] == ')' && strcmp(vocab[i].s + 3, vocab[j].s + 3) == 0)) continue;
			double tmp = 0;
			for (int k = 0; k < layer1_size; ++k) tmp += vocab[i].layer1[k] * vocab[j].layer1[k];
			for (int k = 0; k < 10; ++k) if (vocab[i].inner[k] < tmp) {
				for (int z = N - 1; z > k; --z) {
					vocab[i].inner[z] = vocab[i].inner[z - 1];
					vocab[i].idx[z] = vocab[i].idx[z - 1];
				}
				vocab[i].inner[k] = tmp;
				vocab[i].idx[k] = j;
				break;
			}
		}
	}
	for (itr = mymap.begin(); itr != mymap.end(); itr++) {
		int que[5], r = 0;
		for (int i = 0; i < 5; ++i) if (itr -> second.a[i] != -1) {
			int dup = 0;
			int now = itr -> second.a[i];
			for (int j = 0; j < r; ++j) {
				for (int k = 0; k < N; ++k) {
					for (int z = 0; z < N; ++z) {
						if (vocab[now].idx[k] == vocab[que[j]].idx[z]) {
							dup++;
							break;
						}
					}
				}
			}
			if (dup < 1) que[r++] = now;
		}
		if (r > 2) {
			for(int i = 0; i < 5; ++i) itr -> second.a[i] = (i < r) ? que[i] : -1;
			outputs[output_num] = itr -> second;
			output_num++;
		}
	}
	sort(outputs, outputs + output_num);
	for (int i = 0; i < output_num; ++i) {
		printf("%s\n", vocab[outputs[i].a[0]].s + 3);
		for (int j = 0; j < 5; ++j) {
			if (outputs[i].a[j] == -1) break;
			int tmp = outputs[i].a[j];
			for (int k = 0; k < N; k++) {
				char* s = vocab[vocab[tmp].idx[k]].s;
				if (strlen(s) > 3 && s[0] == '(') s += 3;
				printf("%s ", s);
			}
			printf("\n");
		}
	}
	return 0;
}
