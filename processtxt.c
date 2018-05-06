//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 200
#define TEST_LENGTH 1000000
#define layer1_size 100
#define class_number 5

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
	int cn;
	int *point;
	char *word, *code, codelen;
	int vector_index;
	real layer1[layer1_size];
	long long num;
	real rate[class_number + 1];
	long long amb_cnt;
	long long amb_index;
	int is_word;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int debug_mode = 2, window = 5;
int *vocab_hash;
long long vocab_max_size = 2000000, vocab_size = 0;
long long train_words = 0, word_count_actual = 0;
char sentence[100000000];
int header = 0;

real alpha = 0.025, starting_alpha, sample = 1e-3;
real *expTable;
real amb_vec[30000][5][layer1_size];
int amb_size = 0;

int rec[TEST_LENGTH], recrear = 0;

int hs = 1;

real sqr(real x){
	return x * x;
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
	int a = 0, ch;
	while (!feof(fin)) {
		ch = fgetc(fin);
		if (ch == 13) continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
			if (a > 0) {
				if (ch == '\n') ungetc(ch, fin);
				break;
			}
			if (ch == '\n') {
				strcpy(word, (char *)"</s>");
				return;
			} else continue;
		}
		word[a] = ch;
		a++;
		if (a >= MAX_STRING - 1) a--;   // Truncate too long words
	}
	word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
	unsigned long long a, hash = 0;
	for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
	hash = hash % vocab_hash_size;
	return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
	unsigned int hash = GetWordHash(word);
	while (1) {
		if (vocab_hash[hash] == -1) return -1;
		if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
		hash = (hash + 1) % vocab_hash_size;
	}
	return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
	char word[MAX_STRING];
	ReadWord(word, fin);
	if (feof(fin)) return -1;
	return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
	unsigned int hash, length = strlen(word) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
	strcpy(vocab[vocab_size].word, word);
	vocab[vocab_size].cn = 0;
	vocab[vocab_size].num = 0;
	vocab[vocab_size].vector_index = -1;
	vocab[vocab_size].amb_cnt = 0;
	vocab[vocab_size].amb_index = 0;
	vocab[vocab_size].is_word = 1;
	int i = 0, l = strlen(word);
	char ch;
	for (; i < l; ++i) {
		ch = vocab[vocab_size].word[i];
		if ((ch < 'a' || ch > 'z') && (ch < 'A' || ch > 'Z') && ch != '-') {
			vocab[vocab_size].is_word = 0;
			break;
		}
	}
	vocab_size++;
	// Reallocate memory if needed
	if (vocab_size + 2 >= vocab_max_size) {
		vocab_max_size += 100000;
		vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
	}
	hash = GetWordHash(word);
	while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
	vocab_hash[hash] = vocab_size - 1;
	return vocab_size - 1;
}

void ReadVocab() {
	long long a, i, j, k, size, layerSize;
	char word[MAX_STRING];
	double x;

	FILE *fin = fopen(read_vocab_file, "rb");
	if (fin == NULL) {
		printf("Vocabulary file not found\n");
		exit(1);
	}
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	vocab_size = 0;
	fscanf(fin, "%lld %lld", &size, &layerSize);
	for (i = 0; i < size; ++i)
	{
		if (i % 10000 == 0) printf("%c read vocab %lldK", 13, i / 1000); fflush(stdout);
		fgetc(fin);
		ReadWord(word, fin);
		a = AddWordToVocab(word);
		fscanf(fin, "%lld%d", &vocab[a].amb_cnt, &vocab[a].cn);
		for (j = 2; j < 6; ++j) {
			fscanf(fin, "%lf", &x);
			vocab[a].rate[j] = x;
		}
		for (j = 0; j < layerSize; ++j)
		{
			fscanf(fin, "%lf", &x);
			vocab[a].layer1[j] = (float)x;
		}
		if (vocab[a].amb_cnt) {
			for (j = 0; j < vocab[a].amb_cnt; j++) {
				for (k = 0; k < layer1_size; k++) {
					fscanf(fin, "%lf", &x);
					amb_vec[amb_size][j][k] = x;
				}
			}
			amb_size++;
		}
	}

	fclose(fin);
}

void Output() {
	real inner[11];
	long long a, b, cw, word, idx, last_word, sentence_length = 0, sentence_position = 0;
	long long word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
	long long c;
	real sentence_vec[layer1_size];

	FILE *fi = fopen(train_file, "rb");
	if (fi == NULL) return;
	FILE *fo;
	fo = fopen(output_file, "wb");
	real *neu1 = (real *)calloc(layer1_size + 1, sizeof(real));
	while (1) {
		if (sentence_length == 0) {
			while (1) {
				word = ReadWordIndex(fi);
				if (feof(fi)) break;
				if (word == -1) continue;
				word_count++;
				if (word_count % 10000 == 0) {printf("%cprocessed data size %lldK", 13, word_count / 1000); fflush(stdout);}
				if (word == 0) break;
				sen[sentence_length] = word;
				sentence_length++;
				if (sentence_length >= MAX_SENTENCE_LENGTH) break;
			}
			if (feof(fi)) break;
			if (sentence_length == 0) continue;

			cw = 0;
			for (a = 0; a < sentence_length; a++) {
				b = sen[a];
				if (b == -1) continue;
				for (c = 0; c < layer1_size; c++) {
					sentence_vec[c] += vocab[b].layer1[c];
				}
				cw++;
			}
			if (cw)
			{
				real closev = 0;
				for (c = 0; c < layer1_size; c++) closev += sqr(sentence_vec[c]);
				closev = sqrt(closev);
				for (c = 0; c < layer1_size; c++) sentence_vec[c] /= closev;
			}

			sentence_position = 0;
			header = 0;
		}
		if (feof(fi)) {
			break;
		}
		word = sen[sentence_position];
		if (word == -1) continue;		
		for (c = 0; c < layer1_size; c++) neu1[c] = 0;
		if (word < 1 || word >= vocab_size || vocab[word].amb_cnt == 0);
		else {
			cw = 0;
			for (a = 0; a < window * 2 + 1; a++) if (a != window) {
				c = sentence_position - window + a;
				if (c < 0) continue;
				if (c >= sentence_length) continue;
				last_word = sen[c];
				if (last_word == -1) continue;
				cw ++;
				for (c = 0; c < layer1_size; c++) neu1[c] += vocab[last_word].layer1[c];
			}
			if (cw)
			{
				real closev = 0;
				for (c = 0; c < layer1_size; c++) closev += sqr(neu1[c]);
				closev = sqrt(closev);
				for(c = 0; c < layer1_size; c++){
					neu1[c] = neu1[c] / closev + sentence_vec[c];
				}

				idx = vocab[word].amb_index;
				memset(inner, 0, sizeof(inner));
				for (a = 0; a < vocab[word].amb_cnt; a++) for (c = 0; c < layer1_size; ++c) inner[a] += neu1[c] * amb_vec[idx][a][c];
				c = 0;
				for (a = 1; a < vocab[word].amb_cnt; a++) if (inner[a] > inner[c]) c = a;
				sentence[header++] = '(';
				sentence[header++] = '0' + c;
				sentence[header++] = ')';
			}
		}
		if (word > 0 && word < vocab_size) {
			strcpy(sentence + header, vocab[word].word);
			header += strlen(vocab[word].word);
			sentence[header++] = ' ';
			fflush(fo);		
		}
		sentence_position++;
		if (sentence_position >= sentence_length) {
			sentence[header++] = '\n';
			sentence[header++] = 0;
			fputs(sentence, fo);
			header = 0;
			sentence_length = 0;
			continue;
		}
	}
	fclose(fo);
	fclose(fi);
	free(neu1);
}


int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	if (argc == 1) {
		printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse text data from <file> to train the model\n");
		printf("\t-output <file>\n");
		printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
		printf("\t-window <int>\n");
		printf("\t\tSet max skip length between words; default is 5\n");
		printf("\t-debug <int>\n");
		printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
		printf("\t-read-vocab <file>\n");
		printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
		printf("\nExamples:\n");
		printf("./word2vec -train data.txt -output tagdata.txt -read-vocab cal-output-vocab\n\n");
		return 0;
	}
	output_file[0] = 0;
	read_vocab_file[0] = 0;
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
	vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
	if (read_vocab_file[0] != 0) ReadVocab();
	if (output_file[0] != 0) Output();
	return 0;
}
