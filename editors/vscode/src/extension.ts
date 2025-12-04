import * as path from 'path';
import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;

export function activate(context: vscode.ExtensionContext) {
    // Get server path from configuration
    const config = vscode.workspace.getConfiguration('demetrios');
    const serverPath = config.get<string>('serverPath', 'demetrios-lsp');

    // Server options
    const serverOptions: ServerOptions = {
        command: serverPath,
        args: ['--stdio'],
        transport: TransportKind.stdio
    };

    // Client options
    const clientOptions: LanguageClientOptions = {
        documentSelector: [
            { scheme: 'file', language: 'demetrios' }
        ],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.{d,dem}')
        },
        outputChannelName: 'Demetrios Language Server',
        traceOutputChannel: vscode.window.createOutputChannel('Demetrios LSP Trace')
    };

    // Create and start the client
    client = new LanguageClient(
        'demetrios',
        'Demetrios Language Server',
        serverOptions,
        clientOptions
    );

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('demetrios.restartServer', async () => {
            if (client) {
                await client.stop();
                await client.start();
                vscode.window.showInformationMessage('Demetrios language server restarted');
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('demetrios.runFile', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'demetrios') {
                const filePath = editor.document.fileName;
                const terminal = vscode.window.createTerminal('Demetrios');
                terminal.show();
                terminal.sendText(`dc run "${filePath}"`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('demetrios.runFileJit', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'demetrios') {
                const filePath = editor.document.fileName;
                const terminal = vscode.window.createTerminal('Demetrios JIT');
                terminal.show();
                terminal.sendText(`dc run --jit "${filePath}"`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('demetrios.showHir', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'demetrios') {
                const filePath = editor.document.fileName;
                const terminal = vscode.window.createTerminal('Demetrios HIR');
                terminal.show();
                terminal.sendText(`dc dump-hir "${filePath}"`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('demetrios.showHlir', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'demetrios') {
                const filePath = editor.document.fileName;
                const terminal = vscode.window.createTerminal('Demetrios HLIR');
                terminal.show();
                terminal.sendText(`dc dump-hlir "${filePath}"`);
            }
        })
    );

    // Start the client
    client.start();
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
