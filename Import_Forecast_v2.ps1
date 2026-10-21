param(
    [Parameter(Mandatory=$true)][string]$Server,
    [Parameter(Mandatory=$true)][string]$Database,
    [Parameter(Mandatory=$true)][string]$SqlFile,
    [Parameter(Mandatory=$true)][int]$RunId,
    [Parameter(Mandatory=$true)][int]$ExpectedRows
)
$ErrorActionPreference = 'Stop'
$connection = $null
try {
    $builder = New-Object System.Data.SqlClient.SqlConnectionStringBuilder
    # Use connection-string keywords. PowerShell can resolve dot assignments
    # through the builder's dictionary adapter, where DataSource is not a key.
    $builder['Data Source'] = $Server
    $builder['Initial Catalog'] = $Database
    $builder['Integrated Security'] = $true
    $builder['Encrypt'] = $true
    $builder['TrustServerCertificate'] = $true # Local course SQL instance uses a self-signed certificate.
    $builder['Connect Timeout'] = 30
    $connection = New-Object -TypeName System.Data.SqlClient.SqlConnection -ArgumentList $builder.ConnectionString
    $connection.Open()
    $sql = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $SqlFile).Path)
    $batches = [regex]::Split($sql, '(?im)^\s*GO\s*(?:--[^\r\n]*)?$')
    foreach ($batch in $batches) {
        if ([string]::IsNullOrWhiteSpace($batch)) { continue }
        $command = $connection.CreateCommand()
        try {
            $command.CommandText = $batch
            $command.CommandTimeout = 180
            [void]$command.ExecuteNonQuery()
        } finally { $command.Dispose() }
    }
    $check = $connection.CreateCommand()
    try {
        $check.CommandText = 'SELECT COUNT(*) FROM dbo.SALES_FORECAST WHERE ForecastRunID = @RunId;'
        [void]$check.Parameters.Add('@RunId', [System.Data.SqlDbType]::Int)
        $check.Parameters['@RunId'].Value = $RunId
        $rows = [int]$check.ExecuteScalar()
        if ($rows -ne $ExpectedRows) { throw "Expected $ExpectedRows rows in batch $RunId; found $rows." }
        Write-Output "SQL_IMPORT_OK RunID=$RunId Rows=$rows"
    } finally { $check.Dispose() }
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 2
} finally {
    if ($null -ne $connection) { $connection.Dispose() }
}
