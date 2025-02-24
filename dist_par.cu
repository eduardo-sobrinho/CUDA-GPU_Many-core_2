#include <stdio.h>
#include <stdlib.h>

// Macro para checagem de erro das chamadas às funções do CUDA
#define checa_cuda(result) \
    if (result != cudaSuccess) { \
        printf("%s\n", cudaGetErrorString(result)); \
        exit(1); \
    }

char *aloca_sequencia(int n) {
    char *seq;

    seq = (char *) malloc((n + 1) * sizeof(char));
    if (seq == NULL) {
        printf("\nErro na alocação de estruturas\n");
        exit(1);
    }
    return seq;
}

__global__ void inicializa_GPU(int *a, int nLinhas, int mColunas)
{
    int i; // id GLOBAL da thread

    i = blockIdx.x * blockDim.x + threadIdx.x;

    // Inicializa as colunas da 1ª linha
    if (i < mColunas) {
        a[i] = i;
    }

    // Inicializa a 1ª coluna
    if (i < nLinhas) {
       a[i * mColunas] = i;
    }
}

// Kernel executado na GPU por todas as threads de todos os blocos
__global__ void distancia_GPU(int *a, int nLinhas, int mColunas, char *s, char *r, int *d, int deslocamento, int rodadaExt, int tamBloco)
{
    int i; // id GLOBAL da thread

    i = blockIdx.x * blockDim.x + threadIdx.x;

    int min, celulaDiagonal;

    int it = 0;  // Usado para andar com o índice de r
    
    int rodada = 0;
    int thrdIdx = threadIdx.x;

    while(rodada < 2*tamBloco - 1) {
        
        // Se (a thread estiver entre a 1ª e a última coluna do bloco  E  a 2ª e a última coluna da matriz)  E  não estiver após a última linha da matriz, nem do bloco
        if (((rodada - thrdIdx >= 0 && rodada - thrdIdx < tamBloco) && rodada - i + rodadaExt * deslocamento < mColunas-1) && i < nLinhas-1 && thrdIdx < tamBloco) {

            // Se s[i+1] e r[it+1] forem iguais, copia o valor da diagonal; senão, copia o valor da diagonal acrescido de uma unidade
            celulaDiagonal = s[i+1] == r[rodadaExt * deslocamento - blockIdx.x * deslocamento + it + 1] ?
                                            a[rodadaExt * deslocamento - blockIdx.x * deslocamento + i*mColunas - thrdIdx + rodada] :
                                            a[rodadaExt * deslocamento - blockIdx.x * deslocamento + i*mColunas - thrdIdx + rodada] + 1;

            // Mínimo entre a célula diagonal (já calculada) e a célula de cima (acrescida de uma unidade)
            min = celulaDiagonal < a[rodadaExt * deslocamento - blockIdx.x * deslocamento + i*mColunas - thrdIdx + rodada + 1] + 1 ?
                                            celulaDiagonal :
                                            a[rodadaExt * deslocamento - blockIdx.x * deslocamento + i*mColunas - thrdIdx + rodada + 1] + 1;

            // Mínimo entre a célula à esquerda e o mínimo anterior
            if (a[rodadaExt * deslocamento - blockIdx.x * deslocamento + i*mColunas + mColunas - thrdIdx + rodada] + 1 < min) {
                a[rodadaExt * deslocamento - blockIdx.x * deslocamento + i*mColunas + mColunas + 1 - thrdIdx + rodada] = a[rodadaExt * deslocamento - blockIdx.x * deslocamento + i*mColunas + mColunas - thrdIdx + rodada] + 1;
            } else {
                a[rodadaExt * deslocamento - blockIdx.x * deslocamento + i*mColunas + mColunas + 1 - thrdIdx + rodada] = min;
            }
            
            it++;
        }

        rodada++;

        // Sincronização de barreira entre todas as threads do BLOCO
        __syncthreads();
    }

    if (i == 0) {
        *d = a[nLinhas * mColunas - 1];
    }
}

// Programa principal
int main(int argc, char **argv) {
    int nLinhas,
    mColunas,
    nBytes,
    *d_a,  // Vetor (matriz de distância) da GPU (device)
    
    *d_dist,  // Variável da GPU (device) que conterá a última célula da matriz
    h_dist;   // Valor de retorno da última célula da matriz (conterá a distância)
    
    int n,  // Tamanho da sequência s
        m;  // Tamanho da sequência r

    // Sequências s/r de entrada
    char *h_s,
         *h_r,
         *d_s,
         *d_r;

    FILE *arqEntrada;  // Arquivo texto de entrada

    if(argc != 2) {
        printf("O programa foi executado com argumentos incorretos.\n");
        printf("Uso: ./dist_seq <nome arquivo entrada>\n");
        exit(1);
    }

    // Abre arquivo de entrada
    arqEntrada = fopen(argv[1], "rt");

    if (arqEntrada == NULL) {
        printf("\nArquivo texto de entrada não encontrado\n");
        exit(1);
    }

    // Lê tamanho das sequências s e r
    fscanf(arqEntrada, "%d %d", &n, &m);
    n++;
    m++;

    nLinhas = n;
    mColunas = m;
    nBytes = nLinhas * mColunas * sizeof(int);

    // Aloca vetores s e r
    h_s = aloca_sequencia(n);
    h_r = aloca_sequencia(m);

    // Lê sequências do arquivo de entrada
    h_s[0] = ' ';
    h_r[0] = ' ';
    fscanf(arqEntrada, "%s", &(h_s[1]));
    fscanf(arqEntrada, "%s", &(h_r[1]));

    // Fecha arquivo de entrada
    fclose(arqEntrada);

    
    /* Alocação de memória e checagem de erro */

    // Aloca vetor (matriz de distância) na memória global da GPU
    checa_cuda(cudaMalloc((void **)&d_a, nBytes));

    // Aloca variável (distância) na memória global da GPU
    checa_cuda(cudaMalloc((void **)&d_dist, sizeof(int)));
     
    // Aloca vetor (sequência r) na memória global da GPU
    checa_cuda(cudaMalloc((void **)&d_r, m*sizeof(char)));

    // Aloca vetor (sequência s) na memória global da GPU
    checa_cuda(cudaMalloc((void **)&d_s, n*sizeof(char)));
    
    
    cudaEvent_t d_ini, d_fim;
    cudaEventCreate(&d_ini);
    cudaEventCreate(&d_fim);
    cudaEventRecord(d_ini, 0);
    
    // Máximo entre a quantidade de linhas e de colunas
    int n_threads_bloco = n > m ? n : m;
    
    // Se a quantidade pega for maior que 1024, pega o valor 1024, pois esta é a quantidade máxima de threads que cabem em um bloco
    if (n_threads_bloco > 1024) {
        n_threads_bloco = 1024;
    }
    
    // Determina nBlocos em função de mColunas e n_threads_bloco,
    // ou seja, calcula a quantidade de blocos necessária para cobrir todas as linhas
    int nBlocos = (mColunas + n_threads_bloco - 1) / n_threads_bloco;
    
    inicializa_GPU<<<nBlocos, n_threads_bloco>>>(d_a, nLinhas, mColunas);

    // Copia a sequência s do host para a GPU e checa se houve erro
    checa_cuda(cudaMemcpy(d_s, h_s, n*sizeof(char), cudaMemcpyHostToDevice));

    // Copia a sequência r do host para a GPU e checa se houve erro
    checa_cuda(cudaMemcpy(d_r, h_r, m*sizeof(char), cudaMemcpyHostToDevice));
    
    // Host espera GPU terminar de executar
    cudaDeviceSynchronize();
    
    // Define o tamanho do bloco baseado na quantidade de colunas
    int tamBloco = (mColunas-1) <= 1024 ? mColunas-1 : 1024;
    
    int deslocamento = tamBloco;
    nBlocos = 1;
    
    int rodadaExt = 0;
    
    // Total de repetições necessárias para um bloco percorrer a matriz da esquerda para a direita.
    //   Ex: em uma matriz com 3000 colunas, 1 bloco de tamanho 1024 precisará ser chamado 3 vezes para percorrer a matriz da esq p/ dir
    int blocosLinha = (mColunas-1) % tamBloco == 0 ? (mColunas-1) / tamBloco : (mColunas-1) / tamBloco + 1;
    
    // Total de repetições necessárias para o último bloco começar a processar a última porção da matriz.
    //   Ex: em uma matriz com 2000 linhas, serão necessárias 2 rodadas para um bloco de tamanho 1024 iniciar o processamento da última porção da matriz
    int blocosColuna = (nLinhas-1) % tamBloco == 0 ? (nLinhas-1) / tamBloco : (nLinhas-1) / tamBloco + 1;
    
    // Total de repetições para todos os blocos percorrerem a matriz
    int repeticoes = blocosLinha + blocosColuna;
    
    int linhasRestantes = nLinhas;
    
    while (repeticoes-- > 0) {
 
        // Calcula a distância de edição na GPU
        distancia_GPU<<<nBlocos, n_threads_bloco>>>(d_a, nLinhas, mColunas, d_s, d_r, d_dist, deslocamento, rodadaExt, tamBloco);
        
        rodadaExt++;
        
        linhasRestantes = linhasRestantes - n_threads_bloco;
        if (linhasRestantes > 0)
            nBlocos++;
        
        // Host espera GPU terminar de executar
        cudaDeviceSynchronize();
    }

    // Copia a distância (última célula da matriz) para o host
    checa_cuda(cudaMemcpy(&h_dist, d_dist, sizeof(int), cudaMemcpyDeviceToHost));

    cudaEventRecord(d_fim, 0);
    cudaEventSynchronize(d_fim);
    float d_tempo;      // Tempo de execução na GPU em milissegundos
    cudaEventElapsedTime(&d_tempo, d_ini, d_fim);
    cudaEventDestroy(d_ini);
    cudaEventDestroy(d_fim);

    printf("%d\n", h_dist);
    printf("%.2f\n", d_tempo);

    // Libera vetor (matriz de distância) da memória global da GPU
    cudaFree(d_a);

    // Libera vetores da memória global da GPU
    cudaFree(d_s);
    cudaFree(d_r);

    // Libera vetores da memória do host
    free(h_s);
    free(h_r);

    // Libera variável da memória global da GPU
    cudaFree(d_dist);

    return 0;
}