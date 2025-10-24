# TFM-FINAL




RESUMEN ABONADOS

| Columna / Feature                                | Interpretación             | Riesgo de data leakage     | Recomendación                          |
| ------------------------------------------------ | -------------------------- | -------------------------- | -------------------------------------- |
| `FAntiguedad`                                    | Fecha de alta              | Ninguno                    | Mantener solo en cálculo de antigüedad |
| `Sexo`                                           | Género                     | Ninguno                    | OK, usar one-hot                       |
| `Estado` / `Churn`                               | Target                     | N/A                        | Mantener como target                   |
| `AntiguedadAños`                                 | Años de antigüedad         | Ninguno                    | OK                                     |
| `FechaInicioUltimoAbono`                         | Última fecha de abono      | Ninguno si ≤ fecha corte   | OK                                     |
| `TipoUltimoAbono`                                | Tipo de abono              | Ninguno                    | OK                                     |
| `Irregulares_`                                 | Comportamiento del usuario | **Alto** para `BajaFinal*` | Mantener solo `Activo*`                |
| `DiasDesdeUltimoAbono` / `MesesDesdeUltimoAbono` | Tiempo desde último abono  | Ninguno                    | OK                                     |


SERVICIOS

| Bloque                                                   | Qué mide                    | Riesgo de data leakage        | Recomendación               |
| -------------------------------------------------------- | --------------------------- | ----------------------------- | --------------------------- |
| `Total_conceptos_unicos`, `Total_tipos_servicios_unicos` | Diversidad de servicios     | Bajo (si se filtra por fecha) | Mantener, filtrar por fecha |
| One-hot de `Concepto`, `TipoServicio`                    | Uso específico de servicios | Bajo (si se filtra)           | Mantener, filtrar por fecha |
| Suma de one-hot por persona                              | Frecuencia de uso           | Bajo (si se filtra)           | Mantener                    |
| `UsoServiciosExtra`                                      | Indica uso de servicios     | Ninguno                       | Mantener                    |


PAGOS

| Feature / Grupo                                                  | Qué mide                                                            | Riesgo de data leakage                                        | Recomendación                                               |
| ---------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------- |
| `PagoMetálico`, `PagoRecibo`, `PagoTarjeta`, `PagoTransferencia` | Importe total pagado por cliente según forma de pago                | Bajo, porque solo se usan pagos dentro del intervalo temporal | Mantener, útil para caracterizar comportamiento de pago     |
| `TotalCobrado`                                                   | Total pagado por cliente en el periodo                              | Bajo, intervalo temporal controlado                           | Mantener, muy relevante para churn                          |
| `TipoAbono`                                                      | Código del tipo de abono del cliente                                | Bajo, solo datos hasta fecha de corte                         | Mantener, sirve para ver comportamiento de abono            |
| `TipoAbonoInicial`, `TipoAbonoFinal`, `TipoAbonoFrecuente`       | Tipo de abono al inicio, al final y el más común                    | Bajo                                                          | Mantener, permite identificar cambios de patrón             |
| `NumRenovaciones`                                                | Número de pagos/renovaciones hechas por cliente en el periodo       | Bajo                                                          | Mantener, indica consistencia de pagos                      |
| `NumTiposAbono`                                                  | Diversidad de tipos de abono                                        | Bajo                                                          | Mantener                                                    |
| `CambioAbono`                                                    | Booleano: si hubo cambio de abono                                   | Bajo                                                          | Mantener                                                    |
| `MesesDesdeUltimoPago`                                           | Meses desde la última renovación                                    | Bajo                                                          | Mantener, detecta retrasos o abandono incipiente            |
| `MesesDesdePrimerPago`                                           | Meses desde la primera renovación                                   | Bajo                                                          | Mantener                                                    |
| `DuracionRenovacionMeses`                                        | Duración total entre primera y última renovación                    | Bajo                                                          | Mantener                                                    |
| `VarImporteAprox`, `CefVarImporte`                               | Variabilidad de los pagos de cada cliente                           | Bajo                                                          | Mantener, útil para detectar irregularidades en pagos       |
| `NumImpagosReales`                                               | Número de meses sin pagar (cuando no hay cambio de abono)           | Bajo                                                          | Mantener                                                    |
| `NumReactivaciones`                                              | Número de reactivaciones después de impago                          | Bajo                                                          | Mantener                                                    |
| `PagoSinSaltos`                                                  | Indica si el cliente tiene pagos regulares, irregulares o no aplica | Bajo                                                          | Mantener, se puede usar en one-hot para el modelo           |
| `TotalAbono_XXX`                                                 | Total pagado por cada tipo de abono (pivot)                         | Bajo                                                          | Mantener, permite analizar abonos específicos               |
| `TendenciaPago`                                                  | Pendiente del importe pagado en el periodo (regresión lineal)       | Bajo                                                          | Mantener, identifica clientes que pagan menos con el tiempo |
| `TendenciaPagoMismoAbono`                                        | Igual que anterior pero solo clientes sin cambio de abono           | Bajo                                                          | Mantener, más consistente para clientes estables            |
| `TienePagos`                                                     | Booleano que indica si hay pagos registrados                        | Ninguno                                                       | Mantener, puede usarse como feature binaria                 |

ACCESOS

| Feature / Grupo                                                                                                        | Qué mide                                                     | Riesgo de data leakage                | Recomendación                                     |
| ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------- | ------------------------------------------------- |
| `TotalVisitas`                                                                                                         | Número total de visitas del cliente                          | Bajo, solo datos hasta fecha de corte | Mantener, indica compromiso general               |
| `DiasActivo`                                                                                                           | Número de días distintos en los que el cliente ha accedido   | Bajo                                  | Mantener, refleja regularidad                     |
| `VisitasPorSemana`, `VisitasPorMes`                                                                                    | Frecuencia promedio de acceso                                | Bajo                                  | Mantener, útil para segmentación y churn          |
| `MaxVisitasDia`, `VarVisitasMes`                                                                                       | Variabilidad y picos de uso                                  | Bajo                                  | Mantener                                          |
| `DuracionMediaTotal`, `DuracionStdTotal`                                                                               | Tiempo promedio y desviación de estancia por visita          | Bajo                                  | Mantener, indica intensidad de uso                |
| `DuracionMediaUlt90`, `DeltaDuracionUlt90VsTotal`, `PropVisitasLargas`                                                 | Cambios recientes en duración y proporción de visitas largas | Bajo                                  | Mantener, detecta cambios de hábito               |
| `VisitasUlt30`, `VisitasUlt90`, `VisitasUlt180`, `PropUlt90`                                                           | Tendencia reciente de uso                                    | Bajo                                  | Mantener, relevante para churn predictivo         |
| `DiasDesdeUltima`, `DiasHastaPrimera`                                                                                  | Recencia y antigüedad relativa                               | Bajo                                  | Mantener, útil para cohortes                      |
| `StdDiasEntreVisitas`, `FrecuenciaModal`                                                                               | Regularidad de asistencia                                    | Bajo                                  | Mantener                                          |
| `SemanasConVisita`, `SemanasTotales`, `SemanasConUnaVisita`                                                            | Distribución semanal de visitas                              | Bajo                                  | Mantener                                          |
| `MaxRachaSinVisita`, `MaxRachaConVisita`                                                                               | Rachas de ausencia y presencia                               | Bajo                                  | Mantener, útil para detectar abandono o fidelidad |
| `HoraMediaAcceso`, `HoraStdAcceso`                                                                                     | Patrón horario de visitas                                    | Bajo                                  | Mantener                                          |
| `DiaFavorito`, `VarDiasSemana`, `PropFindesemana`                                                                      | Preferencia por días de la semana                            | Bajo                                  | Mantener, puede usarse como categoría o dummy     |
| `EstacionFavorita`, `PropPrimavera/Verano/Otoño/Invierno`                                                              | Preferencia por estaciones                                   | Bajo                                  | Mantener                                          |
| `VisitasFestivos`, `PropVisitasFestivos`, `VisitasFindesemana`, `PropVisitasFindesemana`                               | Comportamiento en días especiales                            | Bajo                                  | Mantener                                          |
| `VisitasCerrado`, `PropVisitasCerrado`                                                                                 | Accesos a días de cierre (probablemente 0)                   | Bajo                                  | Mantener o descartar                              |
| `VisitasRed9a14`, `PropVisitasRed9a14`, `VisitasRed7a20`, `PropVisitasRed7a20`, `VisitasRed7a15`, `PropVisitasRed7a15` | Accesos en horarios reducidos / especiales                   | Bajo                                  | Mantener                                          |
| `PrimeraVisita`, `UltimaVisita`, `TiempoActivoDias`                                                                    | Historial total de actividad                                 | Bajo                                  | Mantener, útil para recencia y cohortes           |
| `VisitasPrimerTrimestre`, `VisitasUltimoTrimestre`                                                                     | Intensidad al inicio y final del periodo                     | Bajo                                  | Mantener                                          |
| `TieneAccesos`                                                                                                         | Booleano que indica si se registraron accesos                | Ninguno                               | Mantener como feature binaria                     |



┌──────────────────────────────┐
│         Usuario/API          │
│  - Datos completos (JSON)    │
│  - IdPersona / lista de IDs  │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│       Endpoint FastAPI       │
│  /predecir/ /predecir_por_id │
│  /predecir_por_ids           │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│ Validación de entradas       │
│  - Columnas esperadas        │
│  - Tipos de datos y bool → 0/1│
│  - Valores no negativos      │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│   Escalado de variables      │
│  - Cargar scaler MLflow      │
│  - Transformar X             │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│ Carga modelo MLflow          │
│  - Mejor run según AUC       │
│  - Obtener run_id, versión   │
│  - Características importantes│
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  Predicción                  │
│  - y_pred (0/1)              │
│  - y_prob (probabilidad)     │
│  - Nivel riesgo categórico   │
│  - Feature importances       │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│ Formato de salida / guardado │
│  - IdPersona / IdUsuario     │
│  - Variables de entrada JSON │
│  - Predicción y probabilidad │
│  - Nivel de riesgo           │
│  - Timestamp de consulta     │
│  - Metadata: endpoint, runID, versión │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│ Almacenamiento               │
│  - CSV (append)              │
│  - Base de datos SQL (opcional) │
│  - Parquet/DeltaLake (escalable) │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│ Auditoría y trazabilidad     │
│  - Histórico de consultas    │
│  - Reproducibilidad exacta   │
│  - Métricas de uso del API   │
└──────────────────────────────┘


flowchart TD
    A[Usuario / API] --> B[Endpoint FastAPI]
    B --> C[Validación de entradas]
    C --> D[Escalado de variables]
    D --> E[Carga modelo MLflow]
    E --> F[Predicción]
    F --> G[Formato de salida / guardado]
    G --> H[Almacenamiento]
    H --> I[Auditoría y trazabilidad]